# # 该脚本旨在将huggingface模型本地化
# from transformers import AutoModel, AutoTokenizer
#
# model_name = "THUDM/glm-4-9b-chat"
# model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
#
# model.save_pretrained('./local_glm4_9b_chat')
# tokenizer.save_pretrained('./local_glm4_9b_chat')

# import pandas as pd
# tirple_file='KG_db/triples/湖南省衡南县毛湾矿区钨矿普查总结报告_triple.csv'
# triple_DF=pd.read_csv(tirple_file,encoding='utf-8')
# print(triple_DF.head())
# for row in triple_DF.iterrows():
#     print(row['0'],row['1'],row['2'])

# from sentence_transformers import SentenceTransformer
# from langchain.embeddings import HuggingFaceEmbeddings
#
# # 选择模型名称
# embedding_model = "openbmb/MiniCPM-Embedding"
#
# # 加载模型
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
# embeddings.client = SentenceTransformer(embeddings.model_name, device='cuda',trust_remote_code=True)
#
# # 保存模型和tokenizer到本地
# local_directory = './local_openbmb_MiniCPM-Embedding'
# # model.save_pretrained(local_directory)
# # print('模型已保存到:', local_directory)
#
# embeddings.client.save_pretrained(local_directory)
#
# # 如果你需要保存tokenizer
# # tokenizer = embeddings.client.tokenizer  # 如果使用的库提供tokenizer对象
# # tokenizer.save_pretrained(local_directory)
#
# print('模型已保存到:', local_directory)
import torch
from sentence_transformers import SentenceTransformer

model_name = "openbmb/MiniCPM-Embedding"
# model_name='./local_openbmb_MiniCPM-Embedding'
model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={ "torch_dtype": torch.float16})

queries = ["中国的首都是哪里？"]
passages = ["北京", "shanghai"]

INSTRUCTION = "Query: "

embeddings_query = model.encode(queries, prompt=INSTRUCTION)
embeddings_doc = model.encode(passages)

scores = (embeddings_query @ embeddings_doc.T)
print(scores.tolist())  # [[0.35365450382232666, 0.18592746555805206]]
print("加载本地模型成功")

local_directory = './local_openbmb_MiniCPM-Embedding'
model.save_pretrained(local_directory)
print('模型已保存到:', local_directory)