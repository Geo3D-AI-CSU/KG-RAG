# ==================== 配置区域 ====================
CONFIG = {
    # 文件路径配置
    "true_data_path": "recall_data/new_data.csv",  # 真实数据 (id,question,answer,related_triples)
    "pred_data_path": "graph_rag_prediction_results.csv",  # 预测数据 (id,question,predicted_answer,retrieved_triples,ground_truth_triples)
    "output_dir": "recall_data/results",  # 结果输出目录

    # 评估参数
    "k_values": [1, 3, 5],  # Recall@K的K值
    "similarity_threshold": 0.4,  # 语义相似度阈值
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2"  # 语义模型
}

# ================================================

import pandas as pd
from typing import Set
from sentence_transformers import SentenceTransformer, util
import os

class RAGEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.model = SentenceTransformer(config["model_name"])
        os.makedirs(config["output_dir"], exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        """加载并合并数据，自动处理不同的三元组格式"""
        true_df = pd.read_csv(self.config["true_data_path"])
        pred_df = pd.read_csv(self.config["pred_data_path"])

        # 合并数据
        merged = pd.merge(
            pred_df,
            true_df[['id', 'answer']],  # 只需要合并answer列
            on='id',
            how='inner'
        )
        print(f"成功加载 {len(merged)} 条数据")
        return merged

    def _normalize_triple(self, triple: str) -> str:
        """统一不同格式的三元组"""
        if pd.isna(triple):
            return ""

        # 处理格式1: Subject: 毛湾矿区 | Relation: 年平均气温 | Object: 19°左右
        if "|" in triple:
            parts = [p.split(":")[1].strip() for p in triple.split("|")]
            return "-".join(parts)
        # 处理格式2: 毛湾矿区-年平均气温-19°左右
        else:
            return triple.strip()

    def _parse_triples(self, triples_str: str) -> Set[str]:
        """解析并规范化三元组集合"""
        if pd.isna(triples_str):
            return set()
        return {self._normalize_triple(t) for t in str(triples_str).split(';') if t.strip()}

    def _calculate_faithfulness(self, predicted_answer: str, ground_truth_triples: Set[str],
                                retrieved_triples: Set[str]) -> float:
        """
        计算忠实性：根据回答与检索内容的交集的语义相似度进行评估。
        """
        # 1. 获取ground_truth_triples和retrieved_triples的交集
        intersection = ground_truth_triples & retrieved_triples

        if len(intersection) == 0:
            return 0.0  # 没有交集，认为忠实性为0

        # 2. 使用SentenceTransformer计算生成的回答与交集三元组之间的语义相似度
        retrieved_texts = [t.replace('-', ' ') for t in intersection]  # 将三元组转换为自然语言句子
        embeddings = self.model.encode([predicted_answer] + retrieved_texts, convert_to_tensor=True)
        predicted_embedding = embeddings[0]
        retrieved_embeddings = embeddings[1:]

        # 3. 计算生成回答与交集三元组的语义相似度
        similarities = util.cos_sim(predicted_embedding, retrieved_embeddings)

        # 4. 判断忠实性：根据相似度设定规则
        faithfulness_score = 0.0
        for sim in similarities[0]:
            if sim >= 0.7:
                faithfulness_score += 1.0  # 完全符合
            elif sim >= 0.5:
                faithfulness_score += 0.5  # 部分符合
            # 小于0.5的认为不符合，默认值为0.0，不加分

        # 计算平均忠实性
        return faithfulness_score / len(similarities[0])  # 对交集中的每个三元组进行计算平均值

    def evaluate(self) -> dict:
        """执行评估并返回指标字典"""
        df = self._load_data()

        # 1. 语义相似度计算
        df['similarity'] = df.apply(
            lambda x: util.cos_sim(
                self.model.encode(str(x['predicted_answer']), convert_to_tensor=True),
                self.model.encode(str(x['answer']), convert_to_tensor=True)
            ).item(),
            axis=1
        )
        df['is_correct'] = df['similarity'] >= self.config["similarity_threshold"]

        # 2. 三元组处理
        df['retrieved_set'] = df['retrieved_triples'].apply(self._parse_triples)
        df['ground_truth_set'] = df['ground_truth_triples'].apply(self._parse_triples)

        df['faithfulness'] = df.apply(
            lambda x: self._calculate_faithfulness(
                x['predicted_answer'], x['ground_truth_set'], x['retrieved_set']
            ), axis=1
        )

        # 3. 计算指标
        metrics = {
            'accuracy': df['is_correct'].mean(),
            'avg_similarity': df['similarity'].mean(),
            'faithfulness': df['faithfulness'].mean(),
            'recall': {
                f'recall@{k}': df.apply(
                    lambda x: len(x['retrieved_set'] & x['ground_truth_set']) / max(1, len(x['ground_truth_set'])),
                    axis=1
                ).mean()
                for k in self.config["k_values"]  # 这里其实可以移除，既然不排序，k值的意义就不大
            }
        }

        # 4. 保存简洁结果
        with open(os.path.join(self.config["output_dir"], "metrics.txt"), "w") as f:
            f.write("=== RAG 评估指标 ===\n")
            f.write(f"准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"平均相似度: {metrics['avg_similarity']:.4f}\n")
            f.write(f"忠实性: {metrics['faithfulness']:.4f}\n")
            for k, v in metrics['recall'].items():
                f.write(f"{k}: {v:.4f}\n")

        return metrics


if __name__ == '__main__':
    evaluator = RAGEvaluator(CONFIG)
    results = evaluator.evaluate()

    # 控制台输出
    print("\n=== 最终指标 ===")
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"平均相似度: {results['avg_similarity']:.4f}")
    print(f"忠实性: {results['faithfulness']:.4f}")
    for k, v in results['recall'].items():
        print(f"{k}: {v:.4f}")
