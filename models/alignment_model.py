import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentModel(nn.Module):
    def __init__(self, text_model, mol_model, mlm_model=None, hidden_dim=768):
        super(AlignmentModel, self).__init__()
        self.text_model = text_model
        self.mol_model = mol_model
        self.mlm_model = mlm_model
        self.mol_projection = nn.Linear(768, hidden_dim)
        # 添加 text_embeddings 的投影层，将其从 609 维投影到 768 维
        self.text_projection = nn.Linear(768, hidden_dim)

    def forward(self, text_inputs, mol_inputs, mlm_inputs=None, labels=None):
        # 获取 BERT 词汇表大小
        vocab_size = self.text_model.config.vocab_size

        # 直接对 2D 张量的每个元素进行检查并替换超出词汇表范围的 token ID
        text_inputs['input_ids'] = torch.where(
            text_inputs['input_ids'] >= vocab_size,
            torch.tensor(100, dtype=text_inputs['input_ids'].dtype).to(text_inputs['input_ids'].device),
            text_inputs['input_ids']
        )

        # 获取 text_embeddings，使用 last_hidden_state 
        text_outputs = self.text_model(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # 获取 [CLS] token 的嵌入

        # 将 text_embeddings 投影到与 mol_embeddings 相同的维度
        text_embeddings = self.text_projection(text_embeddings)

        # 将分子嵌入映射到与文本嵌入相同的维度
        mol_embeddings = self.mol_projection(mol_inputs)

        if self.mlm_model and mlm_inputs is not None and labels is not None:
            mlm_outputs = self.mlm_model(input_ids=mlm_inputs['input_ids'], attention_mask=mlm_inputs['attention_mask'], labels=labels)
            mlm_loss = mlm_outputs.loss
        else:
            mlm_loss = 0

        return text_embeddings, mol_embeddings, mlm_loss

def contrastive_loss(text_embeddings, mol_embeddings, margin=1.0):
    # Debugging: Print shapes before calculating loss
    # print(f"text_embeddings shape (for contrastive loss): {text_embeddings.shape}")
    # print(f"mol_embeddings shape (for contrastive loss): {mol_embeddings.shape}")
    
    cosine_sim = F.cosine_similarity(text_embeddings.unsqueeze(1), mol_embeddings.unsqueeze(0), dim=-1)
    
    positive_pairs = torch.diag(cosine_sim)
    mask = ~torch.eye(cosine_sim.size(0), dtype=bool).to(cosine_sim.device)
    negative_pairs = cosine_sim[mask].view(cosine_sim.size(0), -1)

    positive_loss = F.relu(margin - positive_pairs).mean()
    negative_loss = F.relu(margin + negative_pairs).mean()
    
    loss = positive_loss + negative_loss
    return loss
