from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio


class GLM4StreamingChat:
    def __init__(self, model_path: str = "./local_glm4_9b_chat"):
        """
        初始化GLM4流式聊天模型(兼容旧版vLLM)
        """
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 初始化异步引擎
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 生成参数
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[151329, 151336, 151338]  # GLM4的停止token
        )

        # 对话历史
        self.history = []

    async def stream_chat(self, query: str):
        """
        异步流式对话生成
        """
        # 更新对话历史
        self.history.append({"role": "user", "content": query})

        # 准备模型输入
        prompt = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )

        # 开始流式生成
        stream_generator = self.engine.generate(
            prompt,
            self.sampling_params,
            request_id="chat_request"
        )

        # 逐步返回结果
        full_response = ""
        async for output in stream_generator:
            new_text = output.outputs[0].text[len(full_response):]
            full_response = output.outputs[0].text
            yield new_text

        # 更新对话历史
        self.history.append({"role": "assistant", "content": full_response})


async def interactive_chat():
    """
    交互式聊天测试
    """
    # 初始化聊天实例
    chat = GLM4StreamingChat()

    print("GLM4-9B流式交互测试 (输入'exit'退出)")
    print("=" * 50)

    while True:
        try:
            # 获取用户输入
            query = input("\n用户: ")
            if query.lower() == 'exit':
                break

            # 打印机器人回复前缀
            print("助手: ", end="", flush=True)

            # 流式生成回复
            full_response = ""
            async for chunk in chat.stream_chat(query):
                print(chunk, end="", flush=True)
                full_response += chunk

            # 确保最后换行
            print()

        except KeyboardInterrupt:
            print("\n对话结束")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            break


if __name__ == "__main__":
    # 运行异步交互
    asyncio.run(interactive_chat())