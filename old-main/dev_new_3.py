from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
import shutil
import os
import json
import glob
import re
from openai import OpenAI

import time
class MPM_model:
    def __init__(self,model_config,RAG_config):
        self.max_model_len=model_config['max_model_len']
        self.tp_size=model_config['tp_size']
        self.model_name=model_config['model_name']
        self.tokenizer=AutoTokenizer.from_pretrained(model_config['model_name'],trust_remote_code=True)
        # self.llm=LLM(model=self.model_name,tensor_parallel_size=self.tp_size,max_model_len=self.max_model_len,trust_remote_code=True,enforce_eager=True,device='cuda:0')
        # self.llm = LLM(model=self.model_name, tensor_parallel_size=self.tp_size, max_model_len=self.max_model_len,
        #                trust_remote_code=True, enforce_eager=True)
        self.stop_token_ids=model_config['stop_token_ids']
        self.sampling_params=SamplingParams(temperature=model_config['temperature'],max_tokens=model_config['max_tokens'],stop_token_ids=self.stop_token_ids)


        self.embedding_model_name=RAG_config['embedding_model_name']
        self.kg_filelist=RAG_config['kg_filelist']
        self.embedding_model=sentence_transformers.SentenceTransformer(self.embedding_model_name,trust_remote_code=True,device='cuda:1')
        # self.embedding_model=sentence_transformers.SentenceTransformer(self.embedding_model_name,trust_remote_code=True,device='cuda:0')
        self.embeddings=HuggingFaceEmbeddings()
        self.embeddings.client= self.embedding_model

        self.query=model_config['query']

    def rag_init(self):
        # 指定要搜索的目录
        directory = self.kg_filelist
        # 获取目录下所有 PDF 文件名的完整路径
        filename_list = glob.glob(os.path.join(directory, '*.csv'))
        kg_filelist=filename_list
        demo_data = []
        for kg_file in kg_filelist:
            loader=CSVLoader(kg_file)
            doc=loader.load()
            demo_data.extend(doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        split_docs = text_splitter.split_documents(demo_data)

        db_file="mpm/tep_db"
        persist_directory=db_file
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        db=Chroma.from_documents(split_docs,self.embeddings,persist_directory=db_file)
        return db
    def prompt_init(self,db):
        input_vec=[]
        meta_vec=[]
        for question in self.query:
            similarDocs=db.similarity_search(question,k=10)
            info = ''
            for similarDoc in similarDocs:
                tep_page = similarDoc.page_content
                tep_line = tep_page.replace('\n',' | ').strip()
                info = info + tep_line +'。'
                meta_vec.append(similarDoc.metadata)
            print("*" * 10)
            prompt_begin = "请结合以下事实三元组所组成的背景知识进行回答,并保证后续内容风格一致。"
            prompt_end = "回答要求：- 引用三元组内容时使用`**`加粗。- 只在答案最后附加“引用信息来源”部分，列出原始三元组数据,引用表达访问为：- Subject: xx | Rel: xx | Object: xx。三元组知识库如下："
            info = question + prompt_begin + prompt_end + info
            print(info)
            print("-" * 10)
            input_vec.append(info)
        return input_vec,meta_vec




    def prompt_init_normal(self):
        input_vec = []
        for question in self.query:
            input_vec.append(question)
        return input_vec

    def mpm_normal(self,input_vec):
        res_vec=self.model_generate(input_vec=input_vec)
        result_json = {"data": res_vec}
        return result_json;
    def model_generate(self,input_vec):
        client = OpenAI(api_key="sk-55b1a00f39a547dba8ff44d24dc51e9c",
                        base_url="https://api.deepseek.com")
        res_vec=[]

        # response = client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[
        #         {"role": "system", "content": "你是一个专业贴心的地质领域助手"},
        #         {"role": "user", "content": input_txt},
        #     ],
        #     stream=False
        # )
        # print(response.choices[0].message.content)


        for input_txt in input_vec:
            prompt_input= [
                {"role": "system", "content": "你是一个专业贴心的地质领域助手"},
                {"role": "user", "content": input_txt},
            ]
            outputs = client.chat.completions.create(
                model="deepseek-chat",
                messages=prompt_input,
                stream=False
            )
            # prompt_input = [{'role': "user", 'content': input_txt}]
            # inputs=self.tokenizer.apply_chat_template(prompt_input,tokenize=False,add_generation_prompt=True)
            # outputs=self.llm.generate(prompts=inputs,sampling_params=self.sampling_params)
            print(outputs.choices[0].message.content)
            # print(outputs[0].outputs[0].text)
            res_vec.append(outputs.choices[0].message.content)
        return res_vec

    def json_res(self,res_vec,meta_vec):
        tep_list=[]
        for meta in meta_vec:
            tep_list.append(meta['source'])
        hash_meta=set(tep_list)
        meta_data=[]
        for idx in hash_meta:
            meta_data.append(idx)

        ans=res_vec[0]

        test_vec=[]
        before_reference = re.split(r"\s*引用信息来源\s*[:：]\s*", ans)[0].strip()
        test_vec.append(before_reference)
        print("*"*10,'/n',test_vec)
        reference_section = ans.split("引用信息来源：")[-1].strip()
        triples_csv = []
        for line in reference_section.split("\n"):
            if line.strip().startswith("-"):
                # 去掉前缀和标点，提取内容
                triple = re.sub(r"^-\s*Subject:\s*", "", line).strip()
                triple = triple.replace(" | Rel: ", ",").replace(" | Object: ", ",").replace("。", "")
                triples_csv.append(triple)

        result_json =  {"data": test_vec,'triples':triples_csv,"meta": meta_data}
        return result_json

    def mpm_main(self,input_vec,meta_vec):

        res_vec=self.model_generate(input_vec=input_vec)
        result_json=self.json_res(res_vec,meta_vec)

        return result_json

if __name__ == '__main__':
    model_config_file='config/mpm_model_config.json'
    rag_config_file='config/mpm_rag_config.json'
    with open(model_config_file,'r') as f:
        model_config = json.load(f)
    with open(rag_config_file,'r') as f:
        rag_config = json.load(f)

    T_start=time.time()
    model=MPM_model(model_config,rag_config)
    result=model.mpm_main()
    print(result)
    T_end=time.time()
    print('-------------完成', '程序运行时间：%s秒' % (int(T_end - T_start)))