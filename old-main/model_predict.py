from datapre.pdf_extract import ExtractPDF
from modelfile.model import Model
from prompt.oie_prompt import OIE_prompt
from RAG.tripels_main import RAG_model
import re
import pandas as pd
import json
import ast

class flask_GLM:

    def __init__(self,config_file):
        self.question = ast.literal_eval(config_file['question'])

        with open(config_file['model'], 'r') as f:
            self.model_config=json.load(f)

        with open(config_file['RAG'], 'r') as f:
            self.RAG_config=json.load(f)

        with open(config_file['prompt'], 'r') as f:
            self.prompt_config=json.load(f)

    def triples_predict(self,csvfile):


        chatModel=Model(self.model_config)
        RAGModel=RAG_model(self.RAG_config)
        db_file='/dataset/new'
        RAG_input=RAGModel.RAG_out(self.question,db_file)
        oie=OIE_prompt(self.prompt_config)
        oie_input=[]
        for id_x in RAG_input:
            oie_tep=oie.input_generate_text(id_x)
            print(oie_tep)
            oie_input.append(oie_tep)
        model_out=chatModel.generate_text(oie_input)
        out_csv=[]
        for triple in model_out:
            triple_tep=re.findall(r"Triple\(Subject\('(.*?)'\), Rel\('(.*?)'\), Object\('(.*?)'\)\)", triple)
            out_csv.extend(triple_tep)

        df=pd.DataFrame(out_csv,columns=['Subject','Rel','Object'])
        df.to_csv(csvfile,index=False,encoding='utf-8')

    def triples_predict_meta(self,csvfile):

        chatModel = Model(self.model_config)

        RAGModel = RAG_model(self.RAG_config)
        db_file = '/dataset/new'
        RAG_input,RAG_meta = RAGModel.RAG_meta_out(self.question, db_file)
        oie = OIE_prompt(self.prompt_config)
        oie_input = []
        for id_x in RAG_input:
            oie_tep = oie.input_generate_text(id_x)
            print(oie_tep)
            oie_input.append(oie_tep)
        model_out = chatModel.generate_text(oie_input)
        out_csv = []
        for triple in model_out:
            triple_tep = re.findall(r"Triple\(Subject\('(.*?)'\), Rel\('(.*?)'\), Object\('(.*?)'\)\)", triple)
            out_csv.extend(triple_tep)

        df = pd.DataFrame(out_csv, columns=['Subject', 'Rel', 'Object'])
        df.to_csv(csvfile, index=False, encoding='utf-8')

        tep_list=[]
        for meta in RAG_meta:
            tep_list.append(meta["source"])

        hash_meta=set(tep_list)

        ans_vec=[]
        for idx_meta in hash_meta:
            ans_vec.append(idx_meta)

        return ans_vec


