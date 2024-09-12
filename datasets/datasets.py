import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import faiss
from torch.utils.data import Dataset

class PubMedDataset:
    def __init__(self, file_path, file_type):
        self.file_path = file_path
        self.file_type = file_type
        self.data = self.load_data(file_path, file_type)
        self.index = {}
        self.embeddings = None  
        self.tokenizer = BertTokenizer.from_pretrained('/mnt/nfs-ssd/data/yushuyang/scibert')
        self.model = BertModel.from_pretrained('/mnt/nfs-ssd/data/yushuyang/scibert')
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def load_data(self, file_path, file_type):
        if file_type == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def build_index(self):
        # 构建索引，假设数据有 'article' 和 'abstract' 列
        self.index = {i: {'article': row['article'], 'abstract': row['abstract']}
                      for i, row in self.data.iterrows()}
        return self.index

    def get_text_by_index(self, idx, text_type='abstract'):
        if idx in self.index:
            if text_type in self.index[idx]:
                return self.index[idx][text_type]
            else:
                raise ValueError(f"Invalid text_type '{text_type}'. Must be 'article' or 'abstract'.")
        else:
            raise IndexError(f"Index {idx} not found in dataset")

    def generate_text_embeddings(self):
        """
        生成文本的 BERT 嵌入向量。
        """
        embeddings = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # 获取所有文本并计算其 BERT 嵌入
        for idx in self.index:
            text = self.index[idx]['abstract']
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 的向量
                embeddings.append(cls_embedding.cpu().numpy())

        # 转换为 numpy 数组并存储在类中
        self.embeddings = np.vstack(embeddings)

    def get_text_embeddings(self):
        """
        返回已生成的文本嵌入，如果还没有生成，则调用生成函数。
        """
        if self.embeddings is None:
            self.generate_text_embeddings()
        return self.embeddings


class SMILES_SideEffect_Dataset(Dataset):
    def __init__(self, smiles_list, labels_list, encode_smiles_func, device=None):
        self.smiles_list = smiles_list
        self.labels_list = labels_list
        self.encode_smiles = encode_smiles_func
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        labels = self.labels_list[idx]
        molecule_vector = self.encode_smiles(smiles, device=self.device)
        return molecule_vector.squeeze(0), torch.tensor(labels, dtype=torch.float32)

class MLMDataset(Dataset):
    def __init__(self, texts, molecules, tokenizer, collator, device):
        self.texts = texts
        self.molecules = molecules
        self.tokenizer = tokenizer
        self.collator = collator
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        molecule = self.molecules[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, molecule

    def collate_fn(self, batch):
        texts = [item[0] for item in batch]
        molecules = [item[1] for item in batch]

        # 检查每个 molecule 是否为字典，并包含 'mol2vec' 键
        for idx, molecule in enumerate(molecules):
            if not isinstance(molecule, dict):
                raise TypeError(f"Expected molecule to be a dict, but got {type(molecule)} at index {idx}. Molecule: {molecule}")
            if 'mol2vec' not in molecule:
                raise KeyError(f"'mol2vec' key not found in molecule at index {idx}. Molecule: {molecule}")

        mlm_inputs = self.collator(texts)
        
        # 将每个分子嵌入转换为 tensor
        mol2vec_tensors = torch.stack([torch.tensor(molecule['mol2vec'], device=self.device) for molecule in molecules])

        return mlm_inputs, mol2vec_tensors

