import os
import pandas as pd
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import sys
import numpy as np
from collections import deque
import jieba
from collections import Counter


class KnowledgeGraphRAG:
    def __init__(self, kg_filelist, embeddings, avoid_nodes=None):
        self.kg_filelist = kg_filelist
        self.embeddings = embeddings
        self.avoid_nodes = avoid_nodes if avoid_nodes else []  # 避让节点列表
        self.stopwords = set(["的", "是", "在", "有", "和", "与", "或", "及"])  # 可扩展停用词

    def reconstruct_question(self, question):
        """
        进行问题重构，提取关键内容，提高检索质量。
        """
        words = [w for w in jieba.lcut(question) if w not in self.stopwords]  # 分词并去停用词
        word_freq = Counter(words)  # 计算词频
        keywords = [word for word, freq in word_freq.most_common(5)]  # 选取最重要的5个词
        reconstructed_query = " ".join(keywords)  # 重构问题
        print(f"【重构问题】{reconstructed_query}")  # 观察优化效果
        return reconstructed_query

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

        return triplets, node_embeddings, relation_embeddings

    def retrieve_subgraph(self, question, triplets, node_embeddings, relation_embeddings, max_hops=3):
        """
        基于问题和三元组的联合嵌入相似度检索子图，并扩展到多跳关系。
        """
        # 计算问题与节点和关系的相似度
        question = self.reconstruct_question(question)
        question_embedding = self.embeddings.embed_query(question)
        question_emb = np.array(question_embedding)

        # 初始化队列用于多跳检索
        queue = deque()
        visited_nodes = set()
        visited_triplets = set()

        # 分层阈值设计
        thresholds = [0.65, 0.4, 0.3]  # 一级、二级、三级阈值

        # 第一跳：找到与问题最相关的初始节点和关系
        for triplet in triplets:
            subj, rel, obj = triplet
            # 如果节点在避让列表中，则跳过

            # 计算相似度
            sim_subj = cosine_similarity([question_emb], [node_embeddings[subj]])[0][0]
            sim_rel = cosine_similarity([question_emb], [relation_embeddings[rel]])[0][0]
            sim_obj = cosine_similarity([question_emb], [node_embeddings[obj]])[0][0]

            score = max(sim_subj, sim_rel, sim_obj)
            if subj in self.avoid_nodes :
                score=max(sim_rel,sim_obj)
            if obj in self.avoid_nodes:
                score=max(sim_rel,sim_subj)

            # 一级节点筛选：只要节点或关系的相似度满足一个条件即可
            if score > thresholds[0]:
                queue.append((subj, rel, obj, 0))  # (subject, relation, object, current_hops)
                visited_triplets.add(tuple(triplet))  # 将列表转换为元组
                print(triplet,'------score------------',score)

        # 多跳扩展
        while queue:
            current_subj, current_rel, current_obj, current_hops = queue.popleft()
            if current_hops >= max_hops:
                continue

            # 找到与当前节点相关的三元组
            for triplet in triplets:
                if tuple(triplet) not in visited_triplets and (triplet[0] == current_obj or triplet[2] == current_obj):
                    subj, rel, obj = triplet
                    # 如果节点在避让列表中，则跳过
                    if subj in self.avoid_nodes or obj in self.avoid_nodes:
                        continue
                    # 计算关系相似度
                    sim_rel_new = cosine_similarity([question_emb], [relation_embeddings[rel]])[0][0]
                    sim_subj_new = cosine_similarity([question_emb], [node_embeddings[subj]])[0][0]
                    sim_obj_new = cosine_similarity([question_emb], [node_embeddings[obj]])[0][0]

                    # 根据当前跳数选择阈值
                    current_threshold = thresholds[current_hops + 1] if current_hops + 1 < len(thresholds) else \
                    thresholds[-1]

                    # 二级和三级节点筛选：适度降低阈值
                    if sim_subj_new > current_threshold or sim_rel_new > current_threshold or sim_obj_new > current_threshold:
                        visited_triplets.add(tuple(triplet))  # 将列表转换为元组
                        # 将邻居节点加入队列
                        if subj not in visited_nodes:
                            visited_nodes.add(subj)
                            queue.append((subj, rel, obj, current_hops + 1))
                        if obj not in visited_nodes:
                            visited_nodes.add(obj)
                            queue.append((subj, rel, obj, current_hops + 1))

        return [list(triplet) for triplet in visited_triplets]  # 将元组转换回列表

    def rag_init(self):
        """
        初始化 RAG，加载知识图谱并生成向量数据库。
        """
        triplets, node_embeddings, relation_embeddings = self.load_and_embed_kg()
        return triplets, node_embeddings, relation_embeddings

    def prompt_init_stream(self, question, triplets, node_embeddings, relation_embeddings):
        """
        生成输入提示并流式输出
        """
        subgraph = self.retrieve_subgraph(question, triplets, node_embeddings, relation_embeddings)

        subgraph_text = "。".join([
            f"Subject: {triplet[0]} | Relation: {triplet[1]} | Object: {triplet[2]}"
            for triplet in subgraph
        ])

        prompt_begin = "请结合以下事实三元组所组成的背景知识进行回答,并保证后续内容风格一致。"
        prompt_end = "回答要求：- 引用三元组内容时使用`**`加粗。- 只在答案最后附加“引用信息来源”部分，列出原始三元组数据,引用表达访问为：- Subject: xx | Rel: xx | Object: xx。三元组知识库如下："

        info = question + prompt_begin + prompt_end + subgraph_text + "。"

        print(info)
        return info
        # 流式输出
        # for char in info:
        #     yield char
        #     sys.stdout.write(char)
        #     sys.stdout.flush()
        # yield "\n"




# 加载知识图谱
kg_filelist = ["mpm/test/new/湖南省衡南县毛湾矿区钨矿普查总结报告_triple.csv"]
embedding_model = sentence_transformers.SentenceTransformer('./local_openbmb_MiniCPM-Embedding', trust_remote_code=True,
                                                            device='cuda:1')
embeddings = HuggingFaceEmbeddings()
embeddings.client = embedding_model

# 初始化 RAG（设置避让节点）
avoid_nodes = ["毛湾矿区", "湘南区域"]  # 避让节点列表
rag = KnowledgeGraphRAG(kg_filelist, embeddings, avoid_nodes)
triplets, node_embeddings, relation_embeddings = rag.rag_init()
from openai import OpenAI
client = OpenAI(api_key="sk-55b1a00f39a547dba8ff44d24dc51e9c", base_url="https://api.deepseek.com")

# 交互式问答
print("\n=== 知识图谱问答系统 ===")
print("输入您的问题，或输入 'exit' 退出。\n")

while True:
    user_query = input("请输入您的问题: ")
    if user_query.lower() == "exit":
        print("退出系统...")
        break

    print("\n【回答】")
    input_txt = rag.prompt_init_stream(user_query, triplets, node_embeddings, relation_embeddings)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业贴心的地质领域助手"},
            {"role": "user", "content": input_txt},
        ],
        stream=False
    )
    print(response.choices[0].message.content)





