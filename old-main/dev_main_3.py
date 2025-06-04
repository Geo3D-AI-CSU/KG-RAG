# 严格保持原有嵌入模型配置的简化版
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
import os
import shutil
import json


class DocRAG:
    def __init__(self, rag_config):
        """完全保持您原有的嵌入模型初始化方式"""
        self.embedding_model_name = rag_config['embedding_model_name']
        self.embedding_model = sentence_transformers.SentenceTransformer(
            self.embedding_model_name,
            trust_remote_code=True,
            device='cuda:1'  # 保持您原有的设备设置
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.embeddings.client = self.embedding_model  # 保持您原有的赋值方式

        # 保持您原有的文档路径设置
        self.kg_filelist = 'dataset/湖南省衡南县毛湾矿区钨矿普查总结报告.csv'
        self.db = self._init_vector_db()

    def _init_vector_db(self):
        """保持您原有的数据库初始化逻辑"""
        demo_data = []
        loader = CSVLoader(self.kg_filelist)
        doc = loader.load()
        demo_data.extend(doc)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
        split_docs = text_splitter.split_documents(demo_data)

        db_file = "mpm/tep_db"
        persist_directory = db_file
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        return Chroma.from_documents(
            split_docs,
            self.embeddings,
            persist_directory=db_file
        )

    def retrieve_chunks(self, triples, k=3):
        """
        根据三元组检索文档chunks
        保持您原有的检索逻辑，仅做封装
        """
        chunks = []
        triple_list = [t.strip() for t in triples.split(';') if t.strip()]

        for triple in triple_list:
            similar_docs = self.db.similarity_search(triple, k=k)
            for doc in similar_docs:
                chunk = doc.page_content.replace('\n', ' | ').strip()
                if chunk not in chunks:
                    chunks.append(chunk)
        return chunks


if __name__ == '__main__':
    # 保持您原有的config加载方式
    rag_config_file = 'config/KG_config.json'
    with open(rag_config_file, 'r') as f:
        rag_config = json.load(f)


    # 加载数据（保持您原有的数据路径）
    QA_df = pd.read_csv('recall_data/new_data.csv')

    # 初始化RAG（保持您原有的初始化方式）
    doc_retriever = DocRAG(rag_config)

    results = []
    for idx, row in QA_df.iterrows():
        chunks = doc_retriever.retrieve_chunks(row['related_triples'])

        results.append({
            'id': row['id'],
            'question': row['question'],
            'related_triples': row['related_triples'],
            'retrieved_chunks': '。'.join(chunks)  # 保持您原有的分隔符
        })

    # 保持您原有的输出格式
    pd.DataFrame(results).to_csv("triple_retrieval_results.csv", index=False)