from datasets import load_dataset

dataset = load_dataset("FinGPT/fingpt-fiqa_qa")

def format(entry):
    sys_promt = "You are a helpful, professional financial assistant."
    
    instruction = entry['instruction']
    input = entry['input']
    output = entry['output']
    
    format_text = f"<s>[INST]<<SYS>>\n{sys_promt}\n<</SYS>>\n{instruction}\n{input}[/INST]{output}</s>"
    
    return {"text": format_text}

def prepare_dataset(data):
    format_dataset = data['train'].map(format)
    return format_dataset

