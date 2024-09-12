import torch
import torch.nn as nn

def predict_side_effects(smiles_vector, alignment_model, predictor_model, device, similar_text_indices, pubmed_dataset, text_tokenizer):
    flat_similar_text_indices = similar_text_indices.flatten()
    similar_texts = [pubmed_dataset.get_text_by_index(idx) for idx in flat_similar_text_indices]
    
    text_inputs = text_tokenizer(similar_texts, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
    mol_inputs = smiles_vector.to(device)

    # Repeat mol_inputs to match the batch size of text_inputs
    batch_size = text_inputs['input_ids'].size(0)
    mol_inputs = mol_inputs.expand(batch_size, -1)  # 扩展 mol_inputs 的第一个维度以匹配 text_inputs 的 batch size

    # 调用整个 alignment_model，而不是单独调用 text_model
    text_embeds, mol_embeds, _ = alignment_model(text_inputs=text_inputs, mol_inputs=mol_inputs)

    if text_embeds.size(1) != mol_embeds.size(1):
        raise ValueError(f"Text embeddings dim {text_embeds.size(1)} and Mol embeddings dim {mol_embeds.size(1)} do not match.")
    
    combined_vector = torch.cat((text_embeds, mol_embeds), dim=1).to(device)
    projection_layer = nn.Linear(1536, 768).to(device)
    combined_vector = projection_layer(combined_vector)

    with torch.no_grad():
        output = predictor_model(combined_vector)
        probabilities = torch.sigmoid(output).cpu().numpy()

    return probabilities

def predict_side_effects_without_retriever(smiles_vector, predictor_model, device):
    """
    Predict side effects based only on the SMILES vector and predictor model, 
    without using text embeddings.
    """
    # Move the SMILES vector to the device
    mol_inputs = smiles_vector.to(device)

    # 扩展 mol_inputs 以便与原来的逻辑保持一致，增加一个 batch 维度
    mol_inputs = mol_inputs.unsqueeze(0)  # 扩展为 batch_size 1，变为 (1, 768)

    # 直接使用 mol_inputs 进行推理，不需要 text_embeds
    combined_vector = mol_inputs  # 不需要 projection_layer，这里直接使用 mol_inputs

    with torch.no_grad():
        output = predictor_model(combined_vector)
        probabilities = torch.sigmoid(output).cpu().numpy()

    return probabilities
