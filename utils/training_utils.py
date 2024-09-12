import torch
from models.alignment_model import contrastive_loss
from models.chemBERTa_encoder import encode_smiles
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(predictions, ground_truths, threshold=0.5):
    """Evaluates the model by calculating accuracy, precision, recall, and F1 score."""
    # Convert probabilities to binary predictions (1 if probability >= threshold, else 0)
    binary_predictions = (predictions >= threshold).astype(int)

    # Convert ground truth strings to binary arrays
    side_effects_classes = [
        'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
        'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders',
        'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures',
        'Vascular disorders', 'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders', 'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders',
        'Cardiac disorders', 'Nervous system disorders', 'Injury, poisoning and procedural complications'
    ]

    ground_truth_binary = []
    for ground_truth in ground_truths:
        binary_labels = [1 if cls in ground_truth else 0 for cls in side_effects_classes]
        ground_truth_binary.append(binary_labels)

    # Evaluate using precision, recall, and F1 score
    ground_truth_binary = np.array(ground_truth_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_binary, binary_predictions, average='macro')

    accuracy = accuracy_score(ground_truth_binary, binary_predictions)
    return accuracy, precision, recall, f1

def train_model(model, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        smiles, labels = batch  # 获取 SMILES 和 labels
        mol_embeds = encode_smiles(smiles, device=device)  # 使用 encode_smiles 函数
        mol_embeds = mol_embeds.float()  # 确保 mol_embeds 是 float 类型的张量
        
        # 继续进行模型前向传递和损失计算
        outputs = model(mol_embeds)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


