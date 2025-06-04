import os
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from KnowledgeGraphRAG_1 import KnowledgeGraphRAG
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import json
from typing import Generator, List, Union

class MPM_deepseek_model:

    def __init__(self, model_config, rag_config):
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
        self.query = []

    def prompt_init(self):
        """
        初始化输入提示，基于知识图谱生成输入文本。
        """
        input_vec = []
        meta_vec = []
        for question in self.query:
            # 使用已初始化的 RAG 生成输入提示
            input_txt, subgraph_text = self.model_rag.prompt_init_stream(question)
            input_vec.append(input_txt)
            meta_vec.append('毛湾矿区钨矿总结知识库.csv,川口矿田外围钨成矿规律知识库.csv')

        return input_vec, meta_vec

    def prompt_init_normal(self):
        """
        初始化普通输入提示，不涉及知识图谱。
        """
        input_vec = []
        for question in self.query:
            input_vec.append(question)
        return input_vec

    def model_generate(self, input_vec):
        """
        调用本地模型生成回答。
        """
        res_vec = []
        for input_txt in input_vec:
            try:
                # 调用本地模型生成回答
                res_vec = self.local_model.llm_generate([input_txt])
                # res_vec.append(response)
                print(res_vec)
            except Exception as e:
                print(f"Error in model_generate: {e}")
                res_vec.append("Error: Failed to generate response.")
        return res_vec

    def normal_res(self, input_vec):
        """
        处理普通预测请求的响应。
        """
        res_vec = self.model_generate(input_vec)
        result_json = {"data": res_vec}
        return result_json

    def json_res(self, res_vec, meta_vec):
        """
        处理知识图谱增强预测的响应，提取引用信息并生成 JSON 结果。
        """
        try:
            # 使用 set 去重
            hash_meta = set(meta_vec)
            meta_data = list(hash_meta)

            ans = res_vec[0]
            test_vec = []
            # before_reference = re.split(r"\s*引用信息来源\s*[:：]\s*", ans)[0].strip()
            before_reference = ans.split("引用信息来源")[0].strip()
            test_vec.append(before_reference)

            # 提取引用信息
            reference_section = ans.split("引用信息来源：")[-1].strip()
            triples_csv = []
            for line in reference_section.split("\n"):
                if line.strip().startswith("-"):
                    triple = re.sub(r"^-\s*Subject:\s*", "", line).strip()
                    triple = triple.replace(" | Relation: ", ",").replace(" | Object: ", ",").replace("。", "")
                    triples_csv.append(triple)

            result_json = {"data": test_vec, 'triples': triples_csv, "meta": meta_data}
            return result_json
        except Exception as e:
            print(f"Error in json_res: {e}")
            return {"error": "Failed to process response."}

    def mpm_main(self, input_vec, meta_vec):
        """
        主函数，处理知识图谱增强的预测请求。
        """
        res_vec = self.model_generate(input_vec)
        result_json = self.json_res(res_vec, meta_vec)
        return result_json

    def model_generate_stream(self, input_vec: List[str]) -> Generator[str, None, None]:
        """
        调用本地模型生成回答(流式)
        """
        for input_txt in input_vec:
            try:
                # 调用本地模型生成回答(流式)
                for chunk in self.local_model.llm_generate_stream([input_txt]):
                    yield chunk
            except Exception as e:
                print(f"Error in model_generate_stream: {e}")
                yield "Error: Failed to generate response."

    def mpm_main_stream(self, input_vec: List[str], meta_vec: List[str]) -> Generator[dict, None, None]:
        """
        主函数，处理知识图谱增强的预测请求(流式)
        """
        buffer = ""
        for chunk in self.model_generate_stream(input_vec):
            buffer += chunk
            # 尝试解析中间结果
            try:
                # 这里可以根据你的需求实现部分结果的解析
                # 例如，可以先返回部分生成的文本，最后再处理引用信息
                yield {"data": [buffer], "status": "partial"}
            except Exception as e:
                print(f"Error in streaming processing: {e}")

        # 最终处理完整结果
        result_json = self.json_res([buffer], meta_vec)
        yield result_json


class dev_model:
    def __init__(self):
        """
        初始化本地模型。
        """
        self.llm = LLM(
            model='./local_glm4_9b_chat',
            tensor_parallel_size=1,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True,
            # quantization = "fp8",
        )
        self.tokenizer = AutoTokenizer.from_pretrained('./local_glm4_9b_chat', trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=1024,
            stop_token_ids=[151329, 151336, 151338],
            stream=True
        )

    def llm_generate(self, text_vec):
        """
        使用本地模型生成回答。
        """
        res_vec = []
        for prompt in text_vec:
            try:
                prompt_input = [{'role': "user", 'content': prompt}]
                inputs = self.tokenizer.apply_chat_template(prompt_input, tokenize=False,
                                                            add_generation_prompt=True)
                outputs = self.llm.generate(prompts=inputs, sampling_params=self.sampling_params)
                res_vec.append(outputs[0].outputs[0].text)
                print(outputs[0].outputs[0].text)
            except Exception as e:
                print(f"Error in llm_generate: {e}")
                res_vec.append("Error: Failed to generate response.")
        return res_vec

    def llm_generate_stream(self, text_vec: List[str]) -> Generator[str, None, None]:
        """
        使用本地模型生成回答(流式)
        """
        for prompt in text_vec:
            try:
                prompt_input = [{'role': "user", 'content': prompt}]
                inputs = self.tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=True)

                # 使用流式生成
                outputs = self.llm.generate(prompts=inputs, sampling_params=self.sampling_params)

                # 遍历生成器获取部分结果
                for output in outputs:
                    for chunk in output.outputs:
                        yield chunk.text

            except Exception as e:
                print(f"Error in llm_generate_stream: {e}")
                yield "Error: Failed to generate response."