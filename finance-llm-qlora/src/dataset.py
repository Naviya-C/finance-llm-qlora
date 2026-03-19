def format_entry(entry):
    sys_promt = "You are a helpful, professional financial assistant."
    
    instruction = entry['instruction']
    input = entry['input'] if entry['input'] else ""
    output = entry['output']
    
    format_text = f"<s>[INST]<<SYS>>\n{sys_promt}\n<</SYS>>\n{instruction}\n{input}[/INST]{output}</s>"
    
    return {"text": format_text}

def prepare_dataset(data):
    format_dataset = data['train'].map(format_entry)
    return format_dataset

