import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from hemBERTa import encode_smiles
from model import SideEffectPredictor, train_model, evaluate_model

# 加载和处理数据
def load_and_process_data(file_path):
    # 加载CSV文件
    data = pd.read_csv(file_path)
    
    # 将副作用文本转换为列表
    data['Side Effects'] = data['Side Effects'].apply(ast.literal_eval)
    
    # 使用MultiLabelBinarizer将副作用转换为多标签二进制格式
    mlb = MultiLabelBinarizer(classes=[
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
    ])
    labels = mlb.fit_transform(data['Side Effects'])
    
    return data['SMILES'].tolist(), labels

class SMILES_SideEffect_Dataset(Dataset):
    def __init__(self, smiles_list, labels_list):
        self.smiles_list = smiles_list
        self.labels_list = labels_list
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        labels = self.labels_list[idx]
        molecule_vector = encode_smiles(smiles)
        return molecule_vector.squeeze(0), torch.tensor(labels, dtype=torch.float32)


# 数据集路径
data_path = "/home/yushuyang/ysy/alignment/smiles_side_effect.csv"

# 加载和处理数据
smiles, labels = load_and_process_data(data_path)

# 划分训练集和验证集 (简单示例)
split_index = int(0.8 * len(smiles))
X_train, X_valid = smiles[:split_index], smiles[split_index:]
y_train, y_valid = labels[:split_index], labels[split_index:]

# 定义数据集和数据加载器
train_dataset = SMILES_SideEffect_Dataset(X_train, y_train)
valid_dataset = SMILES_SideEffect_Dataset(X_valid, y_valid)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SideEffectPredictor(input_dim=768, hidden_dim=512, num_classes=27).to(device)
criterion = nn.BCEWithLogitsLoss()  # 使用多标签二分类损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    valid_loss = evaluate_model(model, valid_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'side_effect_predictor.pth')

