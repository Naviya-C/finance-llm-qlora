from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from config import MODEL_NAME, LORA_CONFIG

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit = True, # This is where the QLoRA "Q" is used in our project. It Quantize the model when it load to 4bit(INT4). Quantize happens now -> LoRA happpens.
        device_map = "auto"
    )
    
    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        r = LORA_CONFIG["r"],
        lora_alpha = LORA_CONFIG["lora_alpha"],
        target_modules = LORA_CONFIG["target_modules"],
        lora_dropout = LORA_CONFIG["lora_dropout"],
        task_type = "CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    return model