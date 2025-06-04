from quart import Quart, render_template, jsonify, request, Response
import json
import torch
from model_test_4 import MPM_GLM_model
import asyncio
import time

# 清理 GPU 缓存
torch.cuda.empty_cache()

# 加载配置文件
model_config_file = 'config/deepseek_config.json'
rag_config_file = 'config/KG_config.json'
with open(model_config_file, 'r') as f:
    model_config = json.load(f)
with open(rag_config_file, 'r') as f:
    rag_config = json.load(f)

# 初始化模型
model = MPM_GLM_model(model_config, rag_config)

# 创建 Quart 应用
app = Quart(__name__)


@app.route('/', methods=['GET', 'POST'])
async def root():
    """
    主页面路由，返回聊天界面。
    """
    return await render_template('chat_form.html')


async def stream_response_from_async(async_gen):
    """
    将异步生成器封装成 JSON 格式流式返回：
    格式为 {"data": ["内容"]}\n\n
    """
    async for chunk in async_gen:
        if chunk.strip():  # 过滤空白内容
            json_chunk = json.dumps({"data": [chunk]}) + "\n\n"
            yield json_chunk.encode("utf-8")


# @app.route('/predict', methods=['POST'])
# # async def predict():
# #     try:
# #         data = await request.json
# #         message = data.get("message")
# #         if not message:
# #             return jsonify({"error": "No message provided"}), 400
# #
# #         model.query = message
# #         input_txt, _ = model.prompt_init()
# #         async_gen = model.model_generate(input_txt)
# #
# #         return Response(
# #             stream_response_from_async(async_gen),
# #             content_type='text/event-stream; charset=utf-8'
# #         )
# #
# #     except Exception as e:
# #         print(f"Error in predict: {e}")
# #         return jsonify({"error": "Internal server error"}), 500

@app.route('/predict', methods=['POST'])
async def predict():
    try:
        data = await request.json
        message = data.get("message")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        model.query = message
        input_txt, _ = model.prompt_init()
        async_gen = model.model_generate(input_txt)

        def stream_response_from_async(async_gen):
            async def async_generator():
                async for chunk in async_gen:
                    # 使用 Server-Sent Events 格式返回数据
                    yield f"data: {chunk}\n\n"
            return async_generator()

        return Response(
            stream_response_from_async(async_gen),
            content_type='text/event-stream; charset=utf-8'
        )

    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/ndb_predict', methods=['POST'])
async def ndb_predict():
    try:
        data = await request.json
        message = data.get("message")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        model.query = message
        input_txt = model.prompt_init_normal()
        async_gen = model.model_generate(input_txt)

        return Response(
            stream_response_from_async(async_gen),
            content_type='text/event-stream; charset=utf-8'
        )

    except Exception as e:
        print(f"Error in ndb_predict: {e}")
        return jsonify({"error": "Internal server error"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
