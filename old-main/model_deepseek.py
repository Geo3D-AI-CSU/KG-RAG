import os

import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from KnowledgeGraphRAG import KnowledgeGraphRAG
from openai import OpenAI
import re
import shutil
import os
import json
import glob


class MPM_deepseek_model:

    def __init__(self,model_config,rag_config):
        self.api_key=model_config['api_key']
        self.api_base=model_config['base_url']
        self.model_name= model_config['model_name']

        self.embedding_model_name=rag_config['embedding_model_name']
        self.embedding_model=sentence_transformers.SentenceTransformer(self.embedding_model_name,trust_remote_code=True,device='cuda:1')
        self.embeddings = HuggingFaceEmbeddings()
        self.embeddings.client = self.embedding_model

        self.kg_filelist=rag_config['kg_filelist']
        self.avoid_nodes = ["毛湾矿区", "湘南区域","川口矿区"]  # 避让节点列表

        self.query=[]

    def prompt_init(self):
        input_vec=[]
        meta_vec=[]
        for question in self.query:
            model_rag = KnowledgeGraphRAG(self.kg_filelist, self.embeddings, self.avoid_nodes)
            triplets, node_embeddings, relation_embeddings = model_rag.rag_init()
            input_txt,subgraph_text = model_rag.prompt_init_stream(question, triplets, node_embeddings, relation_embeddings)
            input_vec.append(input_txt)
            meta_vec.append('毛湾矿区钨矿总结知识库.csv,川口矿田外围钨成矿规律知识库.csv')

        return input_vec,meta_vec

    def prompt_init_normal(self):
        input_vec=[]
        for question in self.query:
            input_vec.append(question)
        return input_vec

    def model_generate(self,input_vec):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        res_vec = []
        for input_txt in input_vec:
            prompt_input = [
                {"role": "system", "content": "你是一个专业贴心的地质领域助手"},
                {"role": "user", "content": input_txt},
            ]
            outputs = client.chat.completions.create(
                model="deepseek-chat",
                messages=prompt_input,
                stream=False
            )
            print(outputs.choices[0].message.content)
            res_vec.append(outputs.choices[0].message.content)
        return res_vec

    def normal_res(self,input_vec):
        res_vec=self.model_generate(input_vec)
        result_json={"data":res_vec}
        return result_json

    def json_res(self,res_vec,meta_vec):

        # 使用 set 去重
        hash_meta = set(meta_vec)
        # 将去重后的结果转换回列表
        meta_data = list(hash_meta)
        ans=res_vec[0]
        test_vec = []
        before_reference = re.split(r"\s*引用信息来源\s*[:：]\s*", ans)[0].strip()
        test_vec.append(before_reference)
        print("*" * 10, '/n', test_vec)
        reference_section = ans.split("引用信息来源：")[-1].strip()
        triples_csv = []
        for line in reference_section.split("\n"):
            if line.strip().startswith("-"):
                # 去掉前缀和标点，提取内容
                triple = re.sub(r"^-\s*Subject:\s*", "", line).strip()
                triple = triple.replace(" | Rel: ", ",").replace(" | Object: ", ",").replace("。", "")
                triples_csv.append(triple)

        result_json = {"data": test_vec, 'triples': triples_csv, "meta": meta_data}
        return result_json

    def mpm_main(self,input_vec,meta_vec):

        res_vec=self.model_generate(input_vec=input_vec)
        result_json=self.json_res(res_vec,meta_vec)

        return result_json










