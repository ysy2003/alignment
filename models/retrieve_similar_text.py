import faiss
import torch

def build_faiss_index(text_embeddings):
    """
    构建 FAISS 索引来处理向量检索
    :param text_embeddings: 预先计算好的文本嵌入矩阵 (numpy array of shape [N, D])
    :return: FAISS 索引对象
    """
    # 初始化 FAISS IndexFlatL2 用于 L2 距离检索
    index = faiss.IndexFlatL2(text_embeddings.shape[1])  # D 是文本嵌入的维度
    index.add(text_embeddings)  # 添加所有的文本嵌入到索引
    return index

def retrieve_similar_text(smiles_vector, index, k=5):
    """
    根据 SMILES 向量在 FAISS 索引中检索相似的文本
    :param smiles_vector: SMILES 分子的嵌入向量 (torch tensor or numpy array)
    :param index: FAISS 索引对象
    :param k: 要检索的文本数量
    :return: 检索到的文本的索引 (list of indices)
    """
    smiles_vector_np = smiles_vector.cpu().numpy().astype('float32')  # FAISS 需要 float32 格式
    D, I = index.search(smiles_vector_np, k)  # 返回距离 D 和索引 I
    return I
