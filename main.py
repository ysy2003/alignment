# main.py

import torch
from models.chemBERTa_encoder import encode_smiles
from models.side_effect_predictor import SideEffectPredictor
from alignment.datasets.datasets import PubMedDataset
from models.alignment_model import AlignmentModel
from models.retrieve_similar_text import build_faiss_index, retrieve_similar_text
from models.predict_side_effect import predict_side_effects
from transformers import BertTokenizer, AutoModelForMaskedLM,AutoModel
from models.projection import ProjectionLayer 
from utils.load import load_alignment_model,load_predictor_model
import torch.nn as nn
import numpy as np
import os


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # PubMed dataset
    pubmed_file_path = '/mnt/nfs-ssd/data/yushuyang/pubmed_abstract/document/test-00000-of-00001.parquet'
    pubmed_dataset = PubMedDataset(file_path=pubmed_file_path, file_type='parquet')
    text_index = pubmed_dataset.build_index()

    # 文本 tokenizer
    text_tokenizer = BertTokenizer.from_pretrained('/mnt/nfs-ssd/data/yushuyang/scibert')

    # 自己输入的一个SMILES字符串
    smiles = "CC1CC2C3CC(C4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C)F"
    smiles_vector = encode_smiles(smiles, device=device)

    # 生成并获取文本嵌入
    print(PubMedDataset.__module__)
    text_embeddings = pubmed_dataset.get_text_embeddings()

    # 构建 FAISS 索引
    faiss_index = build_faiss_index(text_embeddings)
    similar_texts_indices = retrieve_similar_text(smiles_vector, faiss_index)

    # 加载对齐模型和副作用预测模型
    alignment_model_path = '/mnt/nfs-ssd/data/yushuyang/alignment/alignment_model_epoch_30.pth'
    predictor_model_path = '/mnt/nfs-ssd/data/yushuyang/side_effect_predictor.pth'
    alignment_model = load_alignment_model(alignment_model_path, device)
    predictor_model = load_predictor_model(predictor_model_path, device)

    # 多个GPU，并行
    if torch.cuda.device_count() > 1:
        alignment_model = nn.DataParallel(alignment_model)
        predictor_model = nn.DataParallel(predictor_model)

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
    
    # 打印预测的副作用
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

    # for i, prob in enumerate(side_effect_predictions[0]):
    #     print(f"{side_effects_classes[i]}: {prob:.4f}")
    # 对副作用概率和类别进行排序，并按概率从大到小输出
    sorted_predictions = sorted(zip(side_effects_classes, side_effect_predictions[0]), key=lambda x: x[1], reverse=True)

    print(f"Predictions for SMILES: {smiles}")
    for side_effect, prob in sorted_predictions:
        print(f"{side_effect}: {prob:.4f}")


if __name__ == "__main__":
    main()
