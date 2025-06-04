import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import deque
import jieba
from collections import Counter
import glob

class KnowledgeGraphRAG:
    def __init__(self, kg_filelist, embeddings, avoid_nodes=None):
        """
        初始化知识图谱 RAG。
        """
        self.kg_filelist = kg_filelist
        self.embeddings = embeddings
        self.avoid_nodes = avoid_nodes if avoid_nodes else []  # 避让节点列表
        self.stopwords = set(["的", "是", "在", "有", "和", "与", "或", "及"])  # 可扩展停用词

        # 在初始化时加载并嵌入知识图谱
        self.triplets, self.node_embeddings, self.relation_embeddings , self.triplets_embedding= self.load_and_embed_kg()

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
        directory = self.kg_filelist
        filename_list = glob.glob(os.path.join(directory, '*.csv'))
        for kg_file in filename_list:
            df = pd.read_csv(kg_file)
            triplets.extend(df.values.tolist())  # 假设 CSV 格式为 [subject, relation, object]

        # 生成节点和关系的嵌入
        nodes = set()
        relations = set()
        triplets_set = set()
        for triplet in triplets:
            nodes.add(triplet[0])  # subject
            nodes.add(triplet[2])  # object
            relations.add(triplet[1])  # relation
            triplet_string = triplet[0]+triplet[1]+triplet[2]
            triplets_set.add(triplet_string)

        # 生成嵌入（节点和关系）
        node_embeddings = {node: self.embeddings.embed_query(node) for node in nodes}
        relation_embeddings = {rel: self.embeddings.embed_query(rel) for rel in relations}
        triplet_embeddings = {triplet_idx:self.embeddings.embed_query(triplet_idx) for triplet_idx in triplets_set}

        return triplets, node_embeddings, relation_embeddings , triplet_embeddings

    def retrieve_subgraph(self, question, max_hops=3, max_candidates_per_hop=10):
        """
        基于问题和三元组的联合嵌入相似度检索子图，并扩展到多跳关系。
        """
        # 计算问题与节点和关系的相似度
        # question = self.reconstruct_question(question)
        question_embedding = self.embeddings.embed_query(question)
        question_emb = np.array(question_embedding)

        # 初始化队列用于多跳检索
        queue = deque()
        visited_nodes = set()
        visited_triplets = set()

        # 分层阈值设计
        thresholds = [0.60, 0.4, 0.3]  # 一级、二级、三级阈值

        # 第一跳：找到与问题最相关的初始节点和关系
        candidate_triplets = []
        for triplet in self.triplets:
            subj, rel, obj = triplet
            # 如果节点在避让列表中，则跳过
            if subj in self.avoid_nodes or obj in self.avoid_nodes:
                continue

            triplet_tep= subj + rel+ obj
            # 计算相似度
            sim_subj = cosine_similarity([question_emb], [self.node_embeddings[subj]])[0][0]
            sim_rel = cosine_similarity([question_emb], [self.relation_embeddings[rel]])[0][0]
            sim_obj = cosine_similarity([question_emb], [self.node_embeddings[obj]])[0][0]
            sim_triple = cosine_similarity([question_emb], [self.triplets_embedding[triplet_tep]])[0][0]

            score = max(sim_subj, sim_rel, sim_obj, sim_triple)

            # 一级节点筛选：只要节点或关系的相似度满足一个条件即可
            if score > thresholds[0]:
                candidate_triplets.append((triplet, score))
                print(f'三元组{triplet}，评分{score}')# 保存三元组及其评分

        # 如果候选三元组数量小于max_candidates_per_hop，按评分排序并扩展
        if len(candidate_triplets) < max_candidates_per_hop:
            candidate_triplets.sort(key=lambda x: x[1], reverse=True)
            candidate_triplets = candidate_triplets[:max_candidates_per_hop]

        # 将选中的三元组加入队列
        for triplet, score in candidate_triplets:
            queue.append((triplet[0], triplet[1], triplet[2], 0))  # (subject, relation, object, current_hops)
            visited_triplets.add(tuple(triplet))  # 将列表转换为元组

        # 多跳扩展
        while queue:
            current_subj, current_rel, current_obj, current_hops = queue.popleft()
            if current_hops >= max_hops:
                continue

            # 找到与当前节点相关的三元组
            candidate_triplets_next_hop = []
            for triplet in self.triplets:
                if tuple(triplet) not in visited_triplets and (triplet[0] == current_obj or triplet[2] == current_obj):
                    subj, rel, obj = triplet
                    # 如果节点在避让列表中，则跳过
                    # if subj in self.avoid_nodes or obj in self.avoid_nodes:
                    #     continue

                    triplet_tep_new = subj + rel + obj
                    # 计算关系相似度
                    sim_rel_new = cosine_similarity([question_emb], [self.relation_embeddings[rel]])[0][0]
                    sim_subj_new = cosine_similarity([question_emb], [self.node_embeddings[subj]])[0][0]
                    sim_obj_new = cosine_similarity([question_emb], [self.node_embeddings[obj]])[0][0]
                    sim_triple_new = cosine_similarity([question_emb], [self.triplets_embedding[triplet_tep_new]])[0][0]


                    # 根据当前跳数选择阈值
                    current_threshold = thresholds[current_hops + 1] if current_hops + 1 < len(thresholds) else thresholds[-1]

                    # 二级和三级节点筛选：适度降低阈值
                    if sim_subj_new > current_threshold or sim_rel_new > current_threshold or sim_obj_new > current_threshold or sim_triple_new > current_threshold:
                        score = max(sim_subj_new, sim_rel_new, sim_obj_new,sim_triple_new)
                        candidate_triplets_next_hop.append((triplet, score))

            # 按评分排序并保留前max_candidates_per_hop个三元组
            candidate_triplets_next_hop.sort(key=lambda x: x[1], reverse=True)
            tep_candidates_per_hop = max_candidates_per_hop - (current_hops*3)
            candidate_triplets_next_hop = candidate_triplets_next_hop[:tep_candidates_per_hop]

            # 将选中的三元组加入队列
            for triplet, score in candidate_triplets_next_hop:
                visited_triplets.add(tuple(triplet))  # 将列表转换为元组
                # 将邻居节点加入队列
                if triplet[0] not in visited_nodes:
                    visited_nodes.add(triplet[0])
                    queue.append((triplet[0], triplet[1], triplet[2], current_hops + 1))
                if triplet[2] not in visited_nodes:
                    visited_nodes.add(triplet[2])
                    queue.append((triplet[0], triplet[1], triplet[2], current_hops + 1))

        return [list(triplet) for triplet in visited_triplets]  # 将元组转换回列表

    def prompt_init_stream(self, question):
        """
        生成输入提示并流式输出。
        """
        subgraph = self.retrieve_subgraph(question)
        subgraph_text = "。".join([
            f"Subject: {triplet[0]} | Relation: {triplet[1]} | Object: {triplet[2]}"
            for triplet in subgraph
        ])

        prompt_begin = f"问题如下：{question} 请结合以下事实三元组所组成的背景知识进行回答总结,并参考下列回答要求："
        prompt_end = ("1.回答分为总结表达和引用表达两部分。"
                      "2.请先根据提供的事实三元组与问题进行分析从而进行回答，这部分内容作为总结表达内容；"
                      "3.请只在总结回答后对与回答内容最匹配的事实三元组进行罗列，这部分内容作为引用表达。具体要求如下："
                      "在总结表达结束后，附加引用标志文本“引用信息来源”，在标志下方列出采用的所有原始三元组数据，要求三元组不能虚构且保持结构完整"
                      "罗列三元组的引用表达形式为：- Subject: xx | Relation: xx | Object: xx。"
                      f"提供的三元组知识库如下：{subgraph}")

        promot_example='''回答示例如下：
川口矿区的蚀变特征主要包括以下几种：
1. 云英岩化：这是川口矿区最发育的蚀变作用之一，与钨矿化有正相关关系，是气成-高温热液活动的主要蚀变作用之一.
2. 硅化：硅化是川口矿区的一种重要蚀变特征，与石英大脉型黑钨矿化相关，是含钨脉石英-多金属硫化物阶段和多种硫化物化的相关蚀变作用。
3. 钾长石化：钾长石化是川口矿区的一种蚀变作用，与钨矿化有正相关关系，是气成-高温热液活动的主要蚀变作用之一.
4. 电气石化：电气石化也是川口矿区的一种蚀变特征，与钨矿化有伴生蚀变作用.
5. 白云母化：白云母化与钨矿化有伴生蚀变作用，是川口矿区的一种蚀变特征.
6. 绢云母化：绢云母化与石英大脉型黑钨矿化相关，是川口矿区的一种蚀变特征.
7. 绿泥石化：绿泥石化是白水地段的蚀变特征.
8. 黄铁矿化：黄铁矿化是白水地段的蚀变特征.
9. 团块状伟晶岩化：团块状伟晶岩化是白水地段的蚀变特征.
10. 黑云母化：黑云母化是白水地段的蚀变特征.
引用信息来源：
- Subject: 毛湾矿区 | Relation: 蚀变作用 | Object: 云英岩化
- Subject: 云英岩化 | Relation: 相关 | Object: 石英大脉型黑钨矿化
- Subject: 毛湾地段 | Relation: 蚀变特征 | Object: 硅化
- Subject: 含钨脉石英—多金属硫化物阶段 | Relation: 围岩蚀变 | Object: 硅化
- Subject: 白水地段 | Relation: 蚀变特征 | Object: 云英岩化
- Subject: 控矿花岗岩 | Relation: 包括 | Object: 川口岩体的白云母花岗岩
- Subject: 气成—高温热液活动 | Relation: 主要蚀变作用 | Object: 钾长石化
- Subject: 钨矿化 | Relation: 伴生蚀变作用 | Object: 钠长石化'''

        # info = prompt_begin + prompt_end +  promot_example
        info = prompt_begin + prompt_end

        # print(info)
        return info, subgraph_text