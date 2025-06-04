import time

from flask import Flask, render_template, jsonify, request
import json
import torch
from model_ds import MPM_deepseek_model
import logging

# 清理 GPU 缓存
torch.cuda.empty_cache()

# 加载全局变量
model_config_file = 'config/deepseek_config.json'
rag_config_file = 'config/KG_config.json'

# 读取配置文件
with open(model_config_file, 'r') as f:
    model_config = json.load(f)
with open(rag_config_file, 'r') as f:
    rag_config = json.load(f)

# 初始化模型
model = MPM_deepseek_model(model_config, rag_config)

KEYWORD_RESPONSES = {
    "矿石类型":{
        "data":['''
岩体型、石英脉型、不整合面型钨矿均为原生矿石。
按照矿石的矿物组成划分，可分为：（1）石英—黑钨矿矿石，（2）石英—白钨矿矿石，（3）石英—云母—长石—黑钨矿矿石，（4）石英—云母—长石—白钨矿矿石，（5）石英—云母—长石—黑钨矿—白钨矿矿石，（6）石英—云母—长石—硫化物—黑钨矿石矿石，（7）石英—云母—长石—硫化物—白钨矿矿石，（8）石英—云母—长石—硫化物—黑钨矿——白钨矿矿石。其中石英—黑钨矿矿石是石英脉型钨矿的主要矿石类型，石英—白钨矿矿石是不整合面型钨矿的主要矿石类型，石英—云母—长石—白钨矿矿石是岩体型钨矿的主要矿石类型。
'''],
        "triples":[],
        "meta":[]
    },
    "控矿因素":{
        "data":['''
（一）成（控）矿地层
成（控）矿有利地层主要为高涧群架枧田组、泥盆系中统跳马涧组。该层位不仅有丰富的成矿物质而且有促进成矿的矿化剂。尤其是高涧群及其更古老的结晶基底地层，是岩体重熔岩浆的物质基础。从地层钨丰度值统计可知，Pt3j和D2t两地层钨丰度极高，为岩体及其晚期成矿流体提供了丰富的钨源及矿化剂。
同时，泥盆系中统跳马涧组D2t下部含砾砂岩、砂岩及其与下伏地层高涧群架枧田组不整合接触面的硅化破碎带、架枧田组砂质板岩、粉砂质岩等脆性岩石易于产生破碎、断裂及（层间）破碎带，形成导矿和容矿构造。
总之，架枧田组板岩在矿田及其外围大多数矿床（点）的钨矿化中主要起着隔挡矿液的逸散而集中向低压扩容空间（岩体内的“Q型”节理及其产生的张、张扭性断裂、岩体外的层间破碎带及断裂构造）运移并与成矿流体交换成矿物质和矿化剂，使之流体成矿物质和矿化剂浓度更高，更有利于成矿。
（二）成（控）矿花岗岩
矿区岩体是钨成矿的有利岩体，尤其是其中的二云母二长花岗岩和白云母二长花岗岩。有丰富的成矿物质和矿化剂。云英岩化白云母花岗岩是重要的成矿地质体，亦是重要的找矿标志。
岩体的成矿有利部位是白云母花岗岩的隆起部位。其隆起部位亦即围岩地层的褶皱隆起，此类隆起易于产生虚脱空间，有利于岩体晚期高温含矿气-液的聚集，且岩体的隆起部位易产生“Q型”原生横向张节理而给矿化流体的灌入留有空间。多次地层褶皱叠加形成的次级小型凹陷和隆起造成岩体顶界面波状起伏，成生较多的长垣状隆起是有利的成矿构造部位。因此，应重寻找岩体的隆起部位（显现的和隐伏的）。
'''],
        "triples":[],
        "meta":[]
    },
    "找矿标志":{
        "data":['''
 一、地质构造标志
（一）岩石标志
区内重要成矿地层岩石为高涧群砂质板岩和泥盆系跳马涧组砂岩、含砾砂岩、砂砾岩；成矿花岗岩是川口岩体的二云母花岗岩、白云母花岗岩，尤其是云英岩化的白云母花岗岩均为成矿有利岩石。
（二）构造标志
（1）在矿田及其外围地区注意寻找次级小的隆起，特别是花岗岩穹窿构造的次级隐伏的隆起部位，此类岩体的穹窿大多是钨矿化的有利部分，可形成富大的花岗岩内带型的钨矿体。
（2）在高涧群组成的次级小隆起边缘其与D2t跳马涧组之间的不整合接触面或与其复合的断裂带，经变质作用，可形成网脉状钨矿化，热变质程度高的地段则矿床规模大。
二、接触变质及气化—热液变质标志
在外接触带沉积变质岩地层中由于岩石的接触变质现象，钨矿化地段一般发育云英岩化、角岩化，钙质岩地层则为矽卡岩化、大理岩化，表明附近或其深部有隐伏花岗岩，若为白云母花岗岩穹窿则钨矿化的可能性较大。
在沉积地层（外接触带）中其线型云英岩化带、石英（硅化）细脉带应注意沿走向追索或考虑下部有否有大脉型的金属硫化物矿化，进而寻找离岩体近的钨矿化；岩体内带应注意寻找Q型原生节理及由其发展而来的张—扭性断裂。
三、矿体部位的判断标志
（一） 矿石矿物共生组合
（1）花岗岩内带型矿体一般由云英岩—白钨矿组合或石英—黑钨矿—黄铁矿组合为矿头相；石英—黑钨矿、白钨矿、辉钼矿、辉铋矿组合为矿中相；石英—白钨矿—硫化物为矿尾相；石英—硫化物，少量白钨矿组合为矿根相。
（2）岩体外接触带型矿体一般由岩体向远离岩体具有上述矿体的分带，即从岩体向上或向外围同样具有上述分带性。
（二）地球化学标志
（1）黑钨矿、白钨矿化学组分及元素对比值
内带型钨矿体：黑钨矿的Fe/Mn比值和白钨矿的Mo/W比值相对高值一般为矿头相，相对低值时为矿尾相。
外带型钨矿体：一般由岩体向上或远离岩体的钨矿体中，黑钨矿、白钨矿的Fe/Mn、Mo/W比值亦有上述递变规律，即比值由高向低递变。
（2）钨矿物及其共生黄铁矿脉石英的成矿温度
内带型矿体：相同矿物一般矿头相成矿温度相对较高，矿尾相则成矿温度相对较低。
外带型矿体：由岩体向上或向外亦具上述相应的变化规律。
因此，依据上述矿体矿石矿物共生组合，白钨矿、黑钨矿化学组分及元素对比值，以及共生矿物的相对成矿温度可大致判定矿体的相对部位，以便指导找矿和评价矿体和矿床。
        '''],
        "triples":[],
        "meta":[]
    }
}
logging.basicConfig(level=logging.INFO)  # 初始化基础配置
logger = logging.getLogger(__name__)     # 创建logger实例

# 创建 Flask 应用
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def root():
    """
    主页面路由，返回聊天界面。
    """
    return render_template('chatgpt_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    处理知识图谱增强的预测请求。
    """
    if request.method == 'POST':
        try:
            data = request.json
            message = data.get("message")
            if not message:
                return jsonify({"error": "No message provided"}), 400

            print(f"Received message: {message}")

            for keyword, fixed_response in KEYWORD_RESPONSES.items():
                if keyword in message:
                    logger.info(f"keyword matched: {keyword}")
                    result = fixed_response
                    time.sleep(16)
                    return jsonify(result)

            # 设置查询并生成输入提示
            model.query = [message]
            input_vec, meta_vec = model.prompt_init()

            # 调用模型生成结果
            result = model.mpm_main(input_vec, meta_vec)
            return jsonify(result)

        except Exception as e:
            print(f"Error in predict: {e}")
            return jsonify({"error": "Internal server error"}), 500

    else:
        return jsonify({"error": "Method not allowed"}), 405

@app.route('/ndb_predict', methods=['POST'])
def ndb_predict():
    """
    处理普通预测请求（不涉及知识图谱）。
    """
    if request.method == 'POST':
        try:
            data = request.json
            message = data.get("message")
            if not message:
                return jsonify({"error": "No message provided"}), 400

            print(f"Received message: {message}")

            # 设置查询并生成输入提示
            model.query = [message]
            input_vec = model.prompt_init_normal()

            # 调用模型生成结果
            result = model.normal_res(input_vec)
            return jsonify(result)

        except Exception as e:
            print(f"Error in ndb_predict: {e}")
            return jsonify({"error": "Internal server error"}), 500

    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=5000, debug=False)