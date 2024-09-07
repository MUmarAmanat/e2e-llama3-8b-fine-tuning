import logging
import sys
import argparse
import torch
import pandas as pd
import os
import torch.nn.functional as F
import numpy as np

from datasets import load_from_disk, Dataset, DatasetDict

from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding
)
from huggingface_hub.hf_api import HfFolder
    

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tesnor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get('logits')

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),
            'accuracy':accuracy_score(predictions,labels)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--dataset_dir", type=str)
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--num_label", type=str)
    
    
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    HfFolder.save_token('')  ## Make sure to use your own hf_api_token
    
    # from huggingface_hub.hf_api import HfFolder
    # HfFolder.save_token('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')
    
    model_name = args.model_name
    lora_config = LoraConfig(r=32, #rank 32,
                         lora_alpha=32, ## LoRA Scaling factor 
                         target_modules= ['q_proj', 'k_proj', 'v_proj', 'o_proj'],  ## The modules(for example, attention blocks) to apply the LoRA update matrices.
                         lora_dropout = 0.05,
                         bias='none',
                         task_type='SEQ_CLS'
    )

    quantization_config = BitsAndBytesConfig(
                                        load_in_4bit = True, # enable 4-bit quantization
                                        bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
                                        bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
                                        bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
                            )

    model = AutoModelForSequenceClassification.from_pretrained(
                                                                model_name,
                                                                quantization_config=quantization_config,
                                                                num_labels=int(args.num_label)
                                                            )

       
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    logger.info(f"\nModel architecture: {model}")
    

    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f'trainable model parameters: {trainable_model_params}\n \
                all model parameters: {all_model_params} \n \
                percentage of trainable model parameters: {(trainable_model_params / all_model_params) * 100} %'


    logger.info(print_number_of_trainable_model_parameters(model))

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    MAX_LEN = 512
    col_to_delete = ['index', 'Job Title']

    def llama_preprocessing_function(examples):
        return tokenizer(examples['Job Description'], truncation=True, max_length=MAX_LEN)


    dataset = load_from_disk(args.dataset_dir)
    tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
    tokenized_datasets = tokenized_datasets.rename_column("Target", "label")
    tokenized_datasets.set_format("torch")

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)


    training_args = TrainingArguments(
                                    output_dir = args.model_dir,
                                    learning_rate = 1e-4,
                                    per_device_train_batch_size = 8,
                                    per_device_eval_batch_size = 8,
                                    num_train_epochs = int(args.epochs),
                                    weight_decay = 0.01,
                                    evaluation_strategy = 'epoch',
                                    save_strategy = 'epoch',
                                    load_best_model_at_end = True,
                                    logging_dir=f"{args.output_data_dir}/logs",
                                    )
    
    trainer = CustomTrainer(
                            model = model,
                            args = training_args,
                            train_dataset = tokenized_datasets['train'],
                            eval_dataset = tokenized_datasets['val'],
                            tokenizer = tokenizer,
                            data_collator = collate_fn,
                            compute_metrics = compute_metrics,
                        )
    

    # train model
    trainer.train()

    # Saves the model to s3
    trainer.save_model(args.model_dir)
