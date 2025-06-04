from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import os
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from KnowledgeGraphRAG_1 import KnowledgeGraphRAG
from transformers import AutoTokenizer
import re

class dev_model:
    def __init__(self):
        '''
        初始化本地模型
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(
            './local_glm4_9b_chat', trust_remote_code=True)

        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=1024,
            stop_token_ids=[151329, 151336, 151338]
        )

        engine_args = AsyncEngineArgs(
            model='./local_glm4_9b_chat',
            tensor_parallel_size=1,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.history = []

    async def llm_stream_generate(self,prompt:str):
        '''
        使用本地模型生成回答
        '''
        self.history.append({'role': "user", 'content': prompt})
        # prompt_input=[{'role': "user", 'content': prompt}]
        inputs=self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_special_tokens=True
        )

        generator = self.engine.generate(
            inputs,
            self.sampling_params,
            request_id='chat_request'
        )

        full_text= ''
        async for output in generator:
            next_chunk = output.outputs[0].text[len(full_text):]
            print(next_chunk)
            full_text += output.outputs[0].text
            yield next_chunk

        self.history.append({"role": "assistant", "content": full_text})

class MPM_GLM_model:
    def __init__(self,model_config,rag_config):
        """
        初始化模型和知识图谱 RAG。
        """
        # 加载嵌入模型
        self.embedding_model_name = rag_config['embedding_model_name']
        self.embedding_model = sentence_transformers.SentenceTransformer(
            self.embedding_model_name, trust_remote_code=True, device='cuda:1'
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.embeddings.client = self.embedding_model

        # 初始化知识图谱 RAG
        self.kg_filelist = rag_config['kg_filelist']
        self.avoid_nodes = ["毛湾矿区", "湘南区域", "川口矿区"]  # 避让节点列表
        self.model_rag = KnowledgeGraphRAG(self.kg_filelist, self.embeddings, self.avoid_nodes)

        # 初始化本地模型
        self.local_model = dev_model()

        # 初始化查询列表
        self.query = ''

    def prompt_init(self):
        """
        初始化输入提示，基于知识图谱生成输入文本。
        """
        meta_vec = []
        input_txt,subgraph_text=self.model_rag.prompt_init_stream(question=self.query)
        meta_vec.append('毛湾矿区钨矿总结知识库.csv,川口矿田外围钨成矿规律知识库.csv')
        return input_txt,meta_vec

    def prompt_init_normal(self):
        """
        初始化普通输入提示，不涉及知识图谱。
        """
        return self.query

    async def model_generate(self,input_txt:str):
        """
        调用本地流式模型生成回答，返回异步生成器。
        """
        try:
            stream_generator=self.local_model.llm_stream_generate(input_txt)

            async for chunk in stream_generator:
                print(chunk,end='',flush=True)
                yield chunk

        except Exception as e:
            print(f"Error in model_generate: {e}")
            yield "Error: Failed to generate response."




