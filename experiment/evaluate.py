import torch
from torch.utils.data import DataLoader
from datasets.datasets import SMILES_SideEffect_Dataset
from models.side_effect_predictor import SideEffectPredictor
from utils.training_utils import evaluate_model
from utils.data_processing import load_and_process_data
from models.chemBERTa_encoder import encode_smiles
import numpy as np
import pandas as pd
import torch
from models.chemBERTa_encoder import encode_smiles
from datasets.datasets import PubMedDataset
from models.alignment_model import AlignmentModel
from models.retrieve_similar_text import build_faiss_index, retrieve_similar_text
from transformers import BertTokenizer, AutoModelForMaskedLM, AutoModel
from models.projection import ProjectionLayer
from utils.load import load_alignment_model, load_predictor_model
from models.predict_side_effect import predict_side_effects
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def evaluate_model(predictions, ground_truths, threshold=0.5):
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

    # 加载新的 CSV 数据集
    data_path = '/home/yushuyang/ysy/alignment/data/sider.csv'
    data = pd.read_csv(data_path)

    # 提取 SMILES 列和所有副作用的标签列
    smiles_list = data['smiles'].tolist()
    side_effects_list = data.drop(columns=['smiles']).values  # 副作用标签作为二进制数组

    # PubMed dataset
    pubmed_file_path = '/mnt/nfs-ssd/data/yushuyang/pubmed_abstract/document/test-00000-of-00001.parquet'
    pubmed_dataset = PubMedDataset(file_path=pubmed_file_path, file_type='parquet')
    text_index = pubmed_dataset.build_index()

    # 文本 tokenizer
    text_tokenizer = BertTokenizer.from_pretrained('/mnt/nfs-ssd/data/yushuyang/scibert')

    # 加载对齐模型和副作用预测模型
    alignment_model_path = '/mnt/nfs-ssd/data/yushuyang/alignment/alignment_model_epoch_30.pth'
    predictor_model_path = '/mnt/nfs-ssd/data/yushuyang/side_effect_predictor.pth'
    alignment_model = load_alignment_model(alignment_model_path, device)
    predictor_model = load_predictor_model(predictor_model_path, device)

    # 多个GPU，并行
    if torch.cuda.device_count() > 1:
        alignment_model = nn.DataParallel(alignment_model)
        predictor_model = nn.DataParallel(predictor_model)

    all_predictions = []
    for smiles in smiles_list:
        smiles_vector = encode_smiles(smiles, device=device)

        # 生成并获取文本嵌入
        text_embeddings = pubmed_dataset.get_text_embeddings()

        # 构建 FAISS 索引
        faiss_index = build_faiss_index(text_embeddings)
        similar_texts_indices = retrieve_similar_text(smiles_vector, faiss_index)

        # 副作用预测
        side_effect_predictions = predict_side_effects(
            smiles_vector=smiles_vector, 
            alignment_model=alignment_model, 
            predictor_model=predictor_model, 
            device=device, 
            similar_text_indices=similar_texts_indices,
            pubmed_dataset=pubmed_dataset,
            text_tokenizer=text_tokenizer
        )
        all_predictions.append(side_effect_predictions[0])

    # 评估模型
    accuracy, precision, recall, f1, auc = evaluate_model(np.array(all_predictions), side_effects_list)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
