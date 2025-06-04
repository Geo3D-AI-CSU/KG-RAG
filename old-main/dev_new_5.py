import os
import pandas as pd
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sys


class KnowledgeGraphRAG:
    def __init__(self, kg_filelist, embeddings):
        self.kg_filelist = kg_filelist
        self.embeddings = embeddings

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

        # 使用嵌入模型生成节点和关系的向量
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

        print(top_nodes)

        subgraph = []
        subnode = []
        for node, _ in top_nodes:
            if node == '毛湾矿区' or node == '湘南区域':
                continue
            for triplet in triplets:
                if triplet[0] == node and triplet[2] not in ['毛湾矿区', '湘南区域']:
                    subnode.append(triplet[2])
                if triplet[2] == node and triplet[0] not in ['毛湾矿区', '湘南区域']:
                    subnode.append(triplet[0])

        for tep_node in subnode:
            for triplet in triplets:
                if triplet[0] == tep_node or triplet[2] == tep_node:
                    subgraph.append(triplet)
            if len(subgraph) > 20:
                break

        return subgraph

    def rag_init(self):
        """
        初始化 RAG，加载知识图谱并生成向量数据库。
        """
        triplets, node_embeddings, _ = self.load_and_embed_kg()
        return triplets, node_embeddings

    def prompt_init_stream(self, question, triplets, node_embeddings):
        """
        逐步生成输入提示并流式输出
        """
        subgraph = self.retrieve_subgraph(question, triplets, node_embeddings)

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
            sys.stdout.write(char)  # 逐字符输出
            sys.stdout.flush()  # 立刻刷新终端，确保内容实时显示
        yield "\n"


# 加载知识图谱
kg_filelist = ["mpm/test/new/湖南省衡南县毛湾矿区钨矿普查总结报告_triple.csv"]

embedding_model = sentence_transformers.SentenceTransformer('./local_openbmb_MiniCPM-Embedding', trust_remote_code=True,
                                                            device='cuda:1')
embeddings = HuggingFaceEmbeddings()
embeddings.client = embedding_model

# 初始化 RAG
rag = KnowledgeGraphRAG(kg_filelist, embeddings)
triplets, node_embeddings = rag.rag_init()

# 交互式问答
print("\n=== 知识图谱问答系统 ===")
print("输入您的问题，或输入 'exit' 退出。\n")

while True:
    user_query = input("请输入您的问题: ")
    if user_query.lower() == "exit":
        print("退出系统...")
        break

    print("\n【回答】")
    for _ in rag.prompt_init_stream(user_query, triplets, node_embeddings):
        pass  # 这里 `yield` 的内容已经实时输出
    print("\n")  # 换行，准备下一个问题
