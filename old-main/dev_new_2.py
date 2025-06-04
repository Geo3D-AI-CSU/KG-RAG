from modelfile.model import Model
from prompt.oie_prompt import OIE_prompt
import re
import pandas as pd
import json
import ast

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
import shutil
import os

class dev_model:

    def __init__(self,config_file):
        self.question = ast.literal_eval(config_file['question'])

        with open(config_file['model'], 'r') as f:
            self.model_config=json.load(f)

        with open(config_file['prompt'], 'r') as f:
            self.prompt_config=json.load(f)

    def triples_predict(self,csvfile,input):
        chatModel = Model(self.model_config)
        oie = OIE_prompt(self.prompt_config)
        oie_input = []
        for id_x in input:
            oie_tep=oie.input_generate_text(id_x)
            print(oie_tep)
            oie_input.append(oie_tep)
        model_out = chatModel.generate_text(oie_input)
        out_csv = []
        for triple in model_out:
            triple_tep=re.findall(r"Triple\(Subject\('(.*?)'\), Rel\('(.*?)'\), Object\('(.*?)'\)\)", triple)
            out_csv.extend(triple_tep)
        df = pd.DataFrame(out_csv, columns=['Subject', 'Rel', 'Object'])
        df.to_csv(csvfile, index=False, encoding='utf-8')

    def rag_out(self,csvfile):
        filepath=csvfile
        demo_data = []
        loader = CSVLoader(filepath)
        doc = loader.load()
        demo_data.extend(doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        split_docs = text_splitter.split_documents(demo_data)

        embedding_model_name = 'openbmb/MiniCPM-Embedding'
        embedding_model = embedding_model_name
        e_model = sentence_transformers.SentenceTransformer(
            embedding_model, trust_remote_code=True, device='cuda:1')
        embeddings = HuggingFaceEmbeddings()
        embeddings.client = e_model
        print('embeddings', embeddings)
        print('embeddings model_name', embedding_model)

        db_file = 'glm_dev/new_db'
        persist_directory = db_file

        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        db = Chroma.from_documents(split_docs, embeddings, persist_directory=db_file)
        question_vec = []
        for question in self.question:
            similarDocs = db.similarity_search(question, k=15)
            info = ''
            for similarDoc in similarDocs:
                # print(similarDoc.page_content)
                # print("*******************")
                tep_meta = similarDoc.page_content
                metadata_line = tep_meta.replace("\n", " | ").strip()
                # print(metadata_line)
                # print("-------------------分界线")
                info = info + metadata_line + "。"
            print(info)
            question_vec.append(info)

        # df = pd.DataFrame(question_vec, columns=['Subject', 'Rel', 'Object'])
        # df.to_csv(csvfile, index=False, encoding='utf-8')
        return question_vec

