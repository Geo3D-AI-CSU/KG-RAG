from flask import Flask,render_template,jsonify,request
import json
import torch
from model_deepseek import MPM_deepseek_model

# 清理 GPU 缓存
torch.cuda.empty_cache()


#加载全局变量
model_config_file='config/deepseek_config.json'
rag_config_file='config/KG_config.json'
with open(model_config_file, 'r') as f:
    model_config = json.load(f)
with open(rag_config_file, 'r') as f:
    rag_config = json.load(f)

model= MPM_deepseek_model(model_config,rag_config)


app = Flask(__name__)

@app.route('/', methods=['get','POST'])
def root():
    return render_template('chatgpt_form.html')


@app.route('/predict', methods=['get','POST'])
def predict():

    if request.method == 'POST':
        data=request.json
        message=data.get("message")
        print(message)

        model.query=[message]
        input_vec,meta_vec=model.prompt_init()
        result= model.mpm_main(input_vec,meta_vec)
        return jsonify(result)

    else:

        print('predict_error')
        error_info={"error":"predict_error"}
        return jsonify(error_info)

@app.route('/ndb_predict', methods=['get','POST'])
def ndb_predict():
    if request.method == 'POST':
        data=request.json
        message=data.get("message")
        print(message)
        model.query=[message]
        input_vec=model.prompt_init_normal()
        result = model.normal_res(input_vec)
        return jsonify(result)

    else:

        print('predict_error')
        error_info={"error":"predict_error"}
        return jsonify(error_info)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)

