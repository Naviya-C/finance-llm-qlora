MODEL_NAME = "meta-llama/Llama-2-7b-hf"

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05
}

TRAINING_CONFIG = {
    "batch_size": 2,
    "epochs": 3,
    "lr": 2e-4,
    "logging_steps": 10
}