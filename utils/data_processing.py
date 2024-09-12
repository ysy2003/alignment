# utils/data_processing.py

import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

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
