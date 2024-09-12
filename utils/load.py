import torch
from transformers import BertTokenizer, AutoModelForMaskedLM,AutoModel
from models.alignment_model import AlignmentModel
from models.side_effect_predictor import SideEffectPredictor

def load_alignment_model(model_path, device):
    text_model = AutoModel.from_pretrained('/mnt/nfs-ssd/data/yushuyang/scibert').to(device)  
    mol_model = AutoModel.from_pretrained('/mnt/nfs-ssd/data/yushuyang/grover').to(device) 
    alignment_model = AlignmentModel(text_model=text_model, mol_model=mol_model, mlm_model=None).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    alignment_model.load_state_dict(state_dict, strict=False)
    
    return alignment_model


def load_predictor_model(model_path, device):
    model = SideEffectPredictor(input_dim=768, hidden_dim=512, num_classes=27).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model