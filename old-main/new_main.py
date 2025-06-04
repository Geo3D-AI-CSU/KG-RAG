from flask import Flask,render_template,jsonify,request
import json
from dev_new_3 import MPM_model

#加载全局变量
model_config_file = 'config/mpm_model_config_test.json'
rag_config_file = 'config/mpm_rag_config.json'
with open(model_config_file, 'r') as f:
    model_config = json.load(f)
with open(rag_config_file, 'r') as f:
    rag_config = json.load(f)
model = MPM_model(model_config, rag_config)
db=model.rag_init()
###flask APP路由相关

app = Flask(__name__)


@app.route('/', methods=['get','POST'])
def root():
    return render_template('chatgpt_form.html')


@app.route('/predict', methods=['get','POST'])
def predict():

    #这个predict函数用于传出文本信息的json输出

    if request.method == 'POST':
        data=request.json
        message=data.get("message")
        print(message)
        model.query=[message]
        input_vec, meta_vec = model.prompt_init(db)
        result = model.mpm_main(input_vec, meta_vec)
        return jsonify(result)



    else:

        print('predict_error')
        error_info={"error":"predict_error"}
        return jsonify(error_info)

@app.route('/mpm', methods=['get','POST'])
def mpm():

    if request.method == 'POST':
        model.query = ["构造背景-本区的主要构造特征是什么？有哪些断裂、褶皱或构造带与成矿可能相关？请分析地质数据中，哪些构造单元对矿化有控制作用"]
        input_vec_mpm, meta_vec_mpm = model.prompt_init(db)
        result_mpm = model.mpm_main(input_vec_mpm,meta_vec_mpm)
        return jsonify(result_mpm)

    else:
        print('mpm_error')
        error_info={"error":"mpm_error"}
        return jsonify(error_info)

@app.route('/ndb_predict', methods=['get','POST'])
def ndb_predict():
    if request.method == 'POST':
        data=request.json
        message=data.get("message")
        print(message)
        model.query=[message]
        input_vec=model.prompt_init_normal()
        result = model.mpm_normal(input_vec)
        return jsonify(result)

    else:

        print('predict_error')
        error_info={"error":"predict_error"}
        return jsonify(error_info)

