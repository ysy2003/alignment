# models/chemberta_encoder.py

import torch
from transformers import AutoTokenizer, AutoModel

def encode_smiles(smiles_str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    local_path = "/mnt/nfs-ssd/data/yushuyang/chemberta"
    chemberta_tokenizer = AutoTokenizer.from_pretrained(local_path)
    chemberta_model = AutoModel.from_pretrained(local_path).to(device)

    inputs = chemberta_tokenizer(smiles_str, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = chemberta_model(**inputs)
    molecule_vector = outputs.last_hidden_state.mean(dim=1)
    
    return molecule_vector

