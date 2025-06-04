CONFIG = {
    # 文件路径配置
    "true_data_path": "recall_data/new_data.csv",  # 真实数据 (id,question,answer,related_triples)
    "pred_data_path": "predicted_ans.csv",  # 预测数据 (id,question,predicted_answer,retrieved_triples,ground_truth_triples)
    # "output_dir": "recall_data/results",  # 结果输出目录

    # 评估参数
    "k_values": [1, 3, 5],  # Recall@K的K值
    "similarity_threshold": 0.55,  # 语义相似度阈值
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2"  # 语义模型
}

import pandas as pd
from typing import Set
from sentence_transformers import SentenceTransformer, util
import os


def evaluate():
    model = SentenceTransformer(CONFIG["model_name"])
    true_df = pd.read_csv(CONFIG["true_data_path"])
    pred_df = pd.read_csv(CONFIG["pred_data_path"])

    df = pd.merge(pred_df, true_df[['id','answer']], how='inner', on='id')

    df['similarity'] = df.apply(
        lambda x: util.cos_sim(
            model.encode(str(x['predicted_answer']), convert_to_tensor=True),
            model.encode(str(x['answer']), convert_to_tensor=True)
        ).item(),
        axis=1
    )
    print(df['similarity'])
    df['is_correct'] = df['similarity'] >= CONFIG["similarity_threshold"]
    accuracy = df['is_correct'].mean()
    print(accuracy)

if __name__ == '__main__':
    evaluate()