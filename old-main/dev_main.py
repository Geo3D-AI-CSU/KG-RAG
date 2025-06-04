#RAG对比实验
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import shutil
import os
import json
import glob
import re
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from KnowledgeGraphRAG_1 import KnowledgeGraphRAG

class dev_model:
    def __init__(self):

        self.llm = LLM(
            model='./local_glm4_9b_chat',
            tensor_parallel_size=1,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True,
            # quantization = "fp8",
        )
        self.tokenizer = AutoTokenizer.from_pretrained('./local_glm4_9b_chat', trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=1024,
            stop_token_ids=[151329, 151336, 151338]
        )
    def llm_generate(self,query):
        prompt_input = [{'role': "user", 'content': query}]
        inputs = self.tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=True)
        outputs = self.llm.generate(prompts=inputs, sampling_params=self.sampling_params)
        ans = outputs[0].outputs[0].text
        print(ans)
        return ans

class graph_rag:
    def __init__(self,rag_config):

        # 加载嵌入模型
        self.embedding_model_name = rag_config['embedding_model_name']
        self.embedding_model = sentence_transformers.SentenceTransformer(
            self.embedding_model_name, trust_remote_code=True, device='cuda:1'
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.embeddings.client = self.embedding_model

        # 初始化知识图谱 RAG
        self.kg_filelist = rag_config['kg_filelist']
        # self.avoid_nodes = ["毛湾矿区", "湘南区域", "川口矿区"]  # 避让节点列表
        self.avoid_nodes = []
        self.model_rag = KnowledgeGraphRAG(self.kg_filelist, self.embeddings, self.avoid_nodes)

    def prompt_init(self,query):

        input_txt,subgraph = self.model_rag.prompt_init_stream(query)
        return input_txt, subgraph

class doc_rag:
    def __init__(self,rag_config):
        # 加载嵌入模型
        self.embedding_model_name = rag_config['embedding_model_name']
        self.embedding_model = sentence_transformers.SentenceTransformer(
            self.embedding_model_name, trust_remote_code=True, device='cuda:1'
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.embeddings.client = self.embedding_model

        self.kg_filelist = 'dataset/湖南省衡南县毛湾矿区钨矿普查总结报告.csv'

    def rag_init(self):

        demo_data = []

        loader=CSVLoader(self.kg_filelist)
        doc = loader.load()
        demo_data.extend(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
        split_docs = text_splitter.split_documents(demo_data)

        db_file = "mpm/tep_db"
        persist_directory = db_file
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        db = Chroma.from_documents(split_docs, self.embeddings, persist_directory=db_file)
        return db

    def prompt_init(self,db,query):

        similarDocs = db.similarity_search(query,k=10)
        info = ''
        for similarDoc in similarDocs:
            tep_page = similarDoc.page_content
            tep_line = tep_page.replace('\n', ' | ').strip()
            info = info + tep_line + '。'

        prompt_begin = f"问题如下：{query} 请结合以下事实所组成的背景知识进行回答总结,背景知识如下 {info} "
        input_txt =  prompt_begin
        return input_txt,info

if __name__ == '__main__':

    QA_df = pd.read_csv('recall_data/new_data.csv')

    model = dev_model()

    results = []

    for idx,row in QA_df.iterrows():
        qid = row['id']
        question = row['question']

        predicted_answer = model.llm_generate(question)

        results.append({
            'id': qid,
            'question': question,
            'predicted_answer': predicted_answer,
            })

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("predicted_ans.csv", index=False)

    # QA_df = pd.read_csv('recall_data/new_data.csv')
    #
    # model = dev_model()
    #
    #
    # rag_config_file = 'config/KG_config.json'
    # with open(rag_config_file, 'r') as f:
    #     rag_config = json.load(f)
    #
    # kg_rag = graph_rag(rag_config)
    #
    # results = []
    #
    # for idx,row in QA_df.iterrows():
    #     qid = row['id']
    #     question = row['question']
    #     ground_truth_triples = row['related_triples']
    #
    #     # 获取RAG提示词和召回三元组
    #     rag_prompt, subgraph = kg_rag.prompt_init(question)
    #
    #     print(subgraph)
    #
    #     predicted_answer = model.llm_generate(rag_prompt)
    #
    #     # 格式化召回三元组
    #     retrieved_triples = "; ".join([
    #         trip for trip in subgraph.split("。")  # 将 subgraph_text 分割成单个三元组字符串
    #     ])
    #
    #     print(retrieved_triples)
    #
    #     results.append({
    #         'id': qid,
    #         'question': question,
    #         'predicted_answer': predicted_answer,
    #         'retrieved_triples': retrieved_triples,
    #         'ground_truth_triples': ground_truth_triples,
    #     })
    #
    # results_df = pd.DataFrame(results)
    # print(results_df)
    # results_df.to_csv("graph_rag_prediction_results.csv", index=False)










