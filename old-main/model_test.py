import time
from typing import Generator
from model_stream import *


def test_streaming_output():
    # 模拟模型配置和RAG配置
    model_config = {}
    rag_config = {
        'embedding_model_name': './local_openbmb_MiniCPM-Embedding',
        'kg_filelist': ['毛湾矿区钨矿总结知识库.csv', '川口矿田外围钨成矿规律知识库.csv']
    }

    # 初始化模型
    print("初始化模型中...")
    model = MPM_deepseek_model(model_config, rag_config)
    print("模型初始化完成")

    # 设置问题
    questions = [
        "毛湾矿区的主要矿产资源是什么？",
        "川口矿田的成矿规律有哪些特点？"
    ]

    # 测试每个问题
    for question in questions:
        print(f"\n提问: {question}")
        model.query = [question]

        # 获取输入提示
        input_vec, meta_vec = model.prompt_init()

        # 开始流式处理
        print("开始流式回答:")
        start_time = time.time()

        # 用于累积完整响应
        full_response = ""

        # 处理每个流式块
        for chunk in model.mpm_main_stream(input_vec, meta_vec):
            if "status" in chunk and chunk["status"] == "partial":
                # 处理部分响应
                partial_text = chunk["data"][0]
                new_content = partial_text[len(full_response):]  # 只获取新增内容
                if new_content:
                    print(new_content, end='', flush=True)
                    full_response = partial_text
            else:
                # 处理最终完整响应
                print("\n\n=== 完整响应 ===")
                print(json.dumps(chunk, indent=2, ensure_ascii=False))

        end_time = time.time()
        print(f"\n回答生成耗时: {end_time - start_time:.2f}秒")


if __name__ == "__main__":
    test_streaming_output()