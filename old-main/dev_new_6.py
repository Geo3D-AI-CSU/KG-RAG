import os
import pandas as pd
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sys
import numpy as np


class KnowledgeGraphRAG:
    def __init__(self, kg_filelist, embeddings):
        self.kg_filelist = kg_filelist
        self.embeddings = embeddings

    def load_and_embed_kg(self):
        """
        加载 CSV 文件中的三元组并生成图嵌入（包含节点和关系）。
        """
        triplets = []
        for kg_file in self.kg_filelist:
            df = pd.read_csv(kg_file)
            triplets.extend(df.values.tolist())  # 假设 CSV 格式为 [subject, relation, object]

        # 生成节点和关系的嵌入
        nodes = set()
        relations = set()
        for triplet in triplets:
            nodes.add(triplet[0])  # subject
            nodes.add(triplet[2])  # object
            relations.add(triplet[1])  # relation

        # 生成嵌入（节点和关系）
        node_embeddings = {node: self.embeddings.embed_query(node) for node in nodes}
        relation_embeddings = {rel: self.embeddings.embed_query(rel) for rel in relations}

        # 为每个三元组生成联合嵌入（Subject + Relation + Object）
        triplet_embeddings = []
        for triplet in triplets:
            subj_emb = node_embeddings[triplet[0]]
            rel_emb = relation_embeddings[triplet[1]]
            obj_emb = node_embeddings[triplet[2]]
            joint_emb = np.concatenate([subj_emb, rel_emb, obj_emb])  # 拼接向量
            triplet_embeddings.append(joint_emb)

        return triplets, node_embeddings, relation_embeddings, triplet_embeddings

    def retrieve_subgraph(self, question, triplets, node_embeddings, relation_embeddings, triplet_embeddings, k=20):
        """
        基于问题和三元组的联合嵌入相似度检索子图。
        """
        # 计算问题与三元组各部分的相似度
        question_embedding = self.embeddings.embed_query(question)
        question_emb = np.array(question_embedding)

        # 计算问题与每个三元组的相似度（动态权重分配）
        triplet_scores = []
        for idx, triplet in enumerate(triplets):
            subj, rel, obj = triplet
            # 获取各部分嵌入
            subj_emb = node_embeddings[subj]
            rel_emb = relation_embeddings[rel]
            obj_emb = node_embeddings[obj]

            # 计算相似度
            sim_subj = cosine_similarity([question_emb], [subj_emb])[0][0]
            sim_rel = cosine_similarity([question_emb], [rel_emb])[0][0]
            sim_obj = cosine_similarity([question_emb], [obj_emb])[0][0]

            # 动态权重：如果问题中包含动词，则增加关系权重
            if "是什么" in question or "有哪些" in question:
                weights = [0.3, 0.5, 0.2]  # 关系权重更高
            else:
                weights = [0.4, 0.3, 0.3]  # 默认权重

            # 综合得分
            score = (sim_subj * weights[0]) + (sim_rel * weights[1]) + (sim_obj * weights[2])
            triplet_scores.append((triplet, score))

        # 按得分排序并选择 top-k 三元组
        sorted_triplets = sorted(triplet_scores, key=lambda x: x[1], reverse=True)[:k]
        subgraph = [item[0] for item in sorted_triplets]

        return subgraph

    def rag_init(self):
        """
        初始化 RAG，加载知识图谱并生成向量数据库。
        """
        triplets, node_embeddings, relation_embeddings, triplet_embeddings = self.load_and_embed_kg()
        return triplets, node_embeddings, relation_embeddings, triplet_embeddings

    def prompt_init_stream(self, question, triplets, node_embeddings, relation_embeddings, triplet_embeddings):
        """
        生成输入提示并流式输出
        """
        subgraph = self.retrieve_subgraph(question, triplets, node_embeddings, relation_embeddings, triplet_embeddings)

        subgraph_text = "。".join([
            f"Subject: {triplet[0]} | Relation: {triplet[1]} | Object: {triplet[2]}"
            for triplet in subgraph
        ])

        prompt_begin = "请结合以下事实三元组所组成的背景知识进行回答,并保证后续内容风格一致。"
        prompt_end = "回答要求：- 引用三元组内容时使用`**`加粗。- 只在答案最后附加“引用信息来源”部分，列出原始三元组数据,引用表达访问为：- Subject: xx | Rel: xx | Object: xx。三元组知识库如下："

        info = question + prompt_begin + prompt_end + subgraph_text + "。"

        # 流式输出
        for char in info:
            yield char
            sys.stdout.write(char)
            sys.stdout.flush()
        yield "\n"


# 加载知识图谱
kg_filelist = ["mpm/test/new/湖南省衡南县毛湾矿区钨矿普查总结报告_triple.csv"]

embedding_model = sentence_transformers.SentenceTransformer('./local_openbmb_MiniCPM-Embedding', trust_remote_code=True,
                                                            device='cuda:1')
embeddings = HuggingFaceEmbeddings()
embeddings.client = embedding_model

# 初始化 RAG
rag = KnowledgeGraphRAG(kg_filelist, embeddings)
triplets, node_embeddings, relation_embeddings, triplet_embeddings = rag.rag_init()

# 交互式问答
print("\n=== 知识图谱问答系统 ===")
print("输入您的问题，或输入 'exit' 退出。\n")

while True:
    user_query = input("请输入您的问题: ")
    if user_query.lower() == "exit":
        print("退出系统...")
        break

    print("\n【回答】")
    for _ in rag.prompt_init_stream(user_query, triplets, node_embeddings, relation_embeddings, triplet_embeddings):
        pass
    print("\n")