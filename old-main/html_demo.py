from model_predict import flask_GLM
import os
import json
import pandas as pd


def predict_config_read(json_file,question):

    with open(json_file,'r') as f:
        predict_config = json.load(f)

    predict_config['question'] = question
    return predict_config

def model_predict(json_input):

    ans=flask_GLM(json_input)
    result_csv='outputfile/tep_res.csv'
    ans.triples_predict(result_csv)


    DF=pd.read_csv(result_csv)
    triples_DF=DF.sample(n=5,random_state=22)

    triples = [{'subject': row[0], 'Rel': row[1], 'object': row[2]} for index, row in triples_DF.iterrows()]
    # 打包成JSON格式
    result_json = json.dumps({"triples": triples}, ensure_ascii=False)
    print(result_json)

    return result_json


triples_config_file = 'triples_config.json'
question = "['地层','蚀变特征']"
config = predict_config_read(triples_config_file, question)
output = model_predict(config)
print(output)
savepath='test.json'
with open(savepath,'w',encoding='utf-8') as f:
    json.dump(output,f,ensure_ascii=False)