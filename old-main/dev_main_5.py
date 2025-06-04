import os
import glob
import shutil
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import  HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
import numpy as np

class KnowledgeGraphRAG:
    def __init__(self, kg_filelist, embeddings, query):
        self.kg_filelist = kg_filelist
        self.embeddings = embeddings
        self.query = query

    def load_and_embed_kg(self):
        """
        加载 CSV 文件中的三元组并生成图嵌入。
        """
        triplets = []
        for kg_file in self.kg_filelist:
            df = pd.read_csv(kg_file)
            triplets.extend(df.values.tolist())  # 假设 CSV 文件格式为 [subject, relation, object]

        # 生成节点和关系的嵌入
        nodes = set()
        relations = set()
        for triplet in triplets:
            nodes.add(triplet[0])  # subject
            nodes.add(triplet[2])  # object
            relations.add(triplet[1])  # relation

        # 使用预训练的嵌入模型生成节点和关系的向量
        node_embeddings = {node: self.embeddings.embed_query(node) for node in nodes}
        relation_embeddings = {rel: self.embeddings.embed_query(rel) for rel in relations}

        return triplets, node_embeddings, relation_embeddings

    def retrieve_subgraph(self, question, triplets, node_embeddings, k=5):
        """
        根据问题检索相关的子图。
        """
        # 计算问题与节点的相似度
        question_embedding = self.embeddings.embed_query(question)
        node_similarities = {
            node: cosine_similarity([question_embedding], [node_emb])[0][0]
            for node, node_emb in node_embeddings.items()
        }

        # 选择最相关的节点
        top_nodes = sorted(node_similarities.items(), key=lambda x: x[1], reverse=True)[:k]

        # 检索与这些节点相关的三元组
        print('top_nodes----------:',top_nodes)

        subgraph = []
        subnode=[]
        for node, _ in top_nodes:
            if node == '毛湾矿区' or node == '湘南区域':
                continue
            for triplet in triplets:
                if triplet[0] == node and triplet[2] != '毛湾矿区' and triplet[2] != '湘南区域' :
                    subnode.append(triplet[2])
                if triplet[2] == node and triplet[0] != '毛湾矿区' and triplet[0] != '湘南区域':
                    subnode.append(triplet[0])

        for tep_node in subnode:
            if node == '毛湾矿区' or node == '湘南区域':
                continue
            for triplet in triplets:
                if triplet[0] == tep_node or triplet[2] == tep_node :
                    subgraph.append(triplet)
            if len(subgraph)>15:
                break

        return subgraph

    def rag_init(self):
        """
        初始化 RAG，加载知识图谱并生成向量数据库。
        """
        # 加载知识图谱并生成嵌入
        triplets, node_embeddings, relation_embeddings = self.load_and_embed_kg()


        return  triplets, node_embeddings

    def prompt_init(self, triplets, node_embeddings):
        """
        根据问题和检索到的子图生成输入提示。
        """
        input_vec = []
        meta_vec = []

        for question in self.query:
            # 检索相关的子图
            subgraph = self.retrieve_subgraph(question, triplets, node_embeddings)

            # 将子图转换为文本
            subgraph_text = "。".join([
                f"Subject: {triplet[0]} | Relation: {triplet[1]} | Object: {triplet[2]}"
                for triplet in subgraph
            ])

            print(subgraph_text)

            # 构建输入提示
            prompt_begin = "请结合以下事实三元组所组成的背景知识进行回答,并保证后续内容风格一致。"
            prompt_end = "回答要求：- 引用三元组内容时使用`**`加粗。- 只在答案最后附加“引用信息来源”部分，列出原始三元组数据,引用表达访问为：- Subject: xx | Rel: xx | Object: xx。三元组知识库如下："
            info = question + prompt_begin + prompt_end + subgraph_text + "。"

            input_vec.append(info)

        return input_vec, meta_vec

# 示例用法

kg_filelist = ["mpm/test/new/湖南省衡南县毛湾矿区钨矿普查总结报告_triple.csv"]


embedding_model=sentence_transformers.SentenceTransformer('./local_openbmb_MiniCPM-Embedding',trust_remote_code=True,device='cuda:1')
embeddings=HuggingFaceEmbeddings()
embeddings.client= embedding_model
query = ["毛湾矿区的找矿标志？"]

rag = KnowledgeGraphRAG(kg_filelist, embeddings, query)
triplets, node_embeddings = rag.rag_init()
input_vec, meta_vec = rag.prompt_init( triplets, node_embeddings)

for prompt in input_vec:
    print(prompt)

while True:
    user_query = input("请输入您的问题（输入 'exit' 退出）：")
    if user_query.lower() == "exit":
        break

    rag.query = [user_query]  # 更新查询
    for prompt in rag.prompt_init_stream(triplets, node_embeddings):
        print(prompt)