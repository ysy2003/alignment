import torch
from torch.utils.data import DataLoader
from datasets.datasets import SMILES_SideEffect_Dataset
from models.side_effect_predictor import SideEffectPredictor
from utils.training_utils import evaluate_model
from utils.data_processing import load_and_process_data
from utils.load import load_predictor_model
from models.chemBERTa_encoder import encode_smiles
import numpy as np
import pandas as pd
import torch
from models.chemBERTa_encoder import encode_smiles
from models.predict_side_effect import predict_side_effects,predict_side_effects_without_retriever
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def evaluate_model(predictions, ground_truths, threshold=0.4):
    """Evaluates the model by calculating accuracy, precision, recall, F1 score, and AUC."""
    # Convert probabilities to binary predictions (1 if probability >= threshold, else 0)
    binary_predictions = (predictions >= threshold).astype(int)

    # Evaluate using precision, recall, F1 score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, binary_predictions, average='macro')
    accuracy = accuracy_score(ground_truths, binary_predictions)
    

    # AUC calculation (handle multi-label AUC calculation for each side effect)
    try:
        auc = roc_auc_score(ground_truths, predictions, average='macro')
    except ValueError:
        auc = float('nan')  # Handle edge case where AUC cannot be computed

    return accuracy, precision, recall, f1, auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the SMILES dataset
    data_path = '/home/yushuyang/ysy/alignment/data/sider.csv'
    data = pd.read_csv(data_path)

    # Extract SMILES column and side effect labels
    smiles_list = data['smiles'].tolist()
    side_effects_list = data.drop(columns=['smiles']).values  # Side effect labels as binary arrays

    # Load the side effect prediction model
    predictor_model_path = '/mnt/nfs-ssd/data/yushuyang/side_effect_predictor.pth'
    predictor_model = load_predictor_model(predictor_model_path, device)

    # Parallelize over multiple GPUs if available
    if torch.cuda.device_count() > 1:
        predictor_model = nn.DataParallel(predictor_model)

    all_predictions = []
    for smiles in smiles_list:
        # Encode the SMILES string into a vector
        smiles_vector = encode_smiles(smiles, device=device)

        # Predict side effects using the predictor model
        side_effect_predictions = predict_side_effects_without_retriever(
            smiles_vector=smiles_vector, 
            predictor_model=predictor_model, 
            device=device
        )
        all_predictions.append(side_effect_predictions[0])

    # Evaluate the model performance
    accuracy, precision, recall, f1, auc = evaluate_model(np.array(all_predictions), side_effects_list)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
