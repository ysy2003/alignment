import torch
from torch.utils.data import DataLoader
from datasets.datasets import MLMDataset
from models.alignment_model import AlignmentModel
from utils.training_utils import train_model, evaluate_model
from transformers import BertTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import numpy as np
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    chebi_data_dir = "/mnt/nfs-ssd/data/yushuyang/chebi_20"
    X_train = np.load(os.path.join(chebi_data_dir, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(chebi_data_dir, 'y_train.npy'), allow_pickle=True)
    X_valid = np.load(os.path.join(chebi_data_dir, 'X_valid.npy'), allow_pickle=True)
    y_valid = np.load(os.path.join(chebi_data_dir, 'X_valid.npy'), allow_pickle=True)

    text_tokenizer = BertTokenizer.from_pretrained('/mnt/nfs-ssd/data/yushuyang/scibert')
    mlm_model = AutoModelForMaskedLM.from_pretrained('/mnt/nfs-ssd/data/yushuyang/scibert').to(device)
    text_model = AutoModelForMaskedLM.from_pretrained('/mnt/nfs-ssd/data/yushuyang/grover').to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=text_tokenizer, mlm=True, mlm_probability=0.15)
    
    train_dataset = MLMDataset(X_train, y_train, text_tokenizer, data_collator, device)
    valid_dataset = MLMDataset(X_valid, y_valid, text_tokenizer, data_collator, device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=valid_dataset.collate_fn)

    alignment_model = AlignmentModel(text_model=text_model, mol_model=mlm_model).to(device)
    optimizer = torch.optim.Adam(alignment_model.parameters(), lr=1e-5)

    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train_model(alignment_model, train_loader, optimizer, device)
        valid_loss = evaluate_model(alignment_model, valid_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    torch.save(alignment_model.state_dict(), 'alignment_model.pth')
    print("Model saved as alignment_model.pth")

if __name__ == "__main__":
    main()
