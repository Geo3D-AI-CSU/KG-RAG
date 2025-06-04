from model_predict import flask_GLM
import json
import pandas as pd
with open('triples_config.json','r') as f:
    demo_config = json.load(f)

demo_config['question']="['地层','蚀变特征']"
demo_config['model']='config/config.json'
demo_config['RAG']='config/RAG_config.json'
demo_config['prompt']='config/oie_config.json'

ans=flask_GLM(demo_config)

result_csv="outputfile/res_new.csv"
ans.triples_predict(result_csv)

DF=pd.read_csv(result_csv)
print(DF)

tep_df=DF.sample(n=5,random_state=0)
print(tep_df)

triples = [{'subject': row[0], 'Rel': row[1], 'object': row[2]} for index, row in tep_df.iterrows()]
# 打包成JSON格式
result_json = json.dumps({"triples": triples}, ensure_ascii=False)
print(result_json)