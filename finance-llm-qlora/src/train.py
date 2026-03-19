from transformers import TrainingArguments #controls training settings
from trl import SFTTrainer #specialized trainer for instruction tuning

from datasets import load_dataset
from dataset import prepare_dataset 
from model import load_model, apply_lora
from config import TRAINING_CONFIG

def train():
    dataset = load_dataset("FinGPT/fingpt-fiqa_qa")
    dataset = prepare_dataset(dataset)
    
    model, tokenizer = load_model()
    qlora_model = apply_lora(model)
    
    training_args = TrainingArguments(
        output_dir = "../model",
        per_device_train_batch_size = TRAINING_CONFIG['batch_size'],
        num_train_epochs = TRAINING_CONFIG['epochs'],
        learning_rate = TRAINING_CONFIG['lr'],
        logging_steps = TRAINING_CONFIG["logging_steps"],
        gradient_accumulation_steps = 4,
        fp16=True,
        save_strategy="epoch"
    )
    
    trainer = SFTTrainer(
        model = qlora_model,
        train_dataset = dataset["train"],
        dataset_text_field = "text",
        tokenizer = tokenizer,
        args = training_args,
        max_seq_length = 512
    )
    
    trainer.train()
    
    trainer.model.save_pretrained("../model/finance-llm")
    tokenizer.save_pretrained("../model/finance-llm")
    
if __name__ == '__main__':
    train()