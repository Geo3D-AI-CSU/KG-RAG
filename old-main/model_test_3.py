from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from typing import AsyncGenerator
import asyncio


class dev_model:
    def __init__(self):
        """
        初始化本地模型(使用AsyncLLMEngine的流式版本)
        """
        engine_args = AsyncEngineArgs(
            model='./local_glm4_9b_chat',
            tensor_parallel_size=1,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.85
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            './local_glm4_9b_chat',
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=1024,
            stop_token_ids=[151329, 151336, 151338]
        )

    async def llm_generate_stream(self, text_vec: list) -> AsyncGenerator[str, None]:
        """
        异步流式生成回答
        :param text_vec: 输入文本列表
        :return: 异步生成器，逐步产生输出
        """
        for prompt in text_vec:
            try:
                prompt_input = [{'role': "user", 'content': prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    prompt_input,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 开始流式生成
                stream_generator = self.engine.generate(
                    inputs,
                    self.sampling_params,
                    request_id=f"chat_{id(prompt)}"  # 唯一请求ID
                )

                # 逐步返回结果
                full_response = ""
                async for output in stream_generator:
                    new_text = output.outputs[0].text[len(full_response):]
                    full_response = output.outputs[0].text
                    yield new_text

            except Exception as e:
                print(f"Error in llm_generate_stream: {e}")
                yield "Error: Failed to generate response."

