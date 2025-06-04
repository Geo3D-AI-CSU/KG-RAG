
from transformers import AutoModel, AutoTokenizer

model_name = "THUDM/glm-4-9b-chat"
model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

model.save_pretrained('./local_glm4_9b_chat')
tokenizer.save_pretrained('./local_glm4_9b_chat')

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
#
# embeddings.client.save_pretrained(local_directory)
#
# # 如果你需要保存tokenizer
# # tokenizer = embeddings.client.tokenizer  # 如果使用的库提供tokenizer对象
# # tokenizer.save_pretrained(local_directory)
#
print('模型已保存到:')
#
# # # 加载本地模型
# # local_embeddings = HuggingFaceEmbeddings(model_name=local_directory)
# # local_embeddings.client = SentenceTransformer(local_embeddings.model_name, device='cuda')
# # print('本地模型加载成功')

# import torch
# from sentence_transformers import SentenceTransformer
#
# model_name = "openbmb/MiniCPM-Embedding"
# # model_name='./local_openbmb_MiniCPM-Embedding'
# model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={ "torch_dtype": torch.float16})
#
#
#
# local_directory = './local_openbmb_MiniCPM-Embedding'
# model.save_pretrained(local_directory)
# print('模型已保存到:', local_directory)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
#
# model_path = "./local_glm4_9b_chat"
#
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
#
# input_text = "请简单介绍钨矿的找矿特征。"
# inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
#
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=100)
#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
