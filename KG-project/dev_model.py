from transformers import AutoTokenizer
from vllm import LLM,SamplingParams

class dev_model:
    def __init__(self):
        self.llm=LLM(
            model='./local_glm4_9b_chat',
            tensor_parallel_size=1,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True,
            # quantization = "fp8",
        )
        self.tokenizer = AutoTokenizer.from_pretrained('./local_glm4_9b_chat',trust_remote_code=True)
        self.sampling_params=SamplingParams(
            temperature=0.95,
            max_tokens=1024,
            stop_token_ids=[151329,151336,151338]
        )

    def llm_generate(self,text_vec):
        # res_vec=[]
        for prompt in text_vec:
            prompt_input=[{'role':"user",'content':prompt}]
            inputs=self.tokenizer.apply_chat_template(prompt_input,tokenize=False,add_generation_prompt=True)
            outputs = self.llm.generate(prompts=inputs, sampling_params=self.sampling_params)
            # res_vec.append(outputs[0].outputs[0].text)
            print(outputs[0].outputs[0].text)
        return outputs[0].outputs[0].text

    def llm_txt_generate(self,text):
        prompt_input = [{'role': "user", 'content': text}]
        inputs = self.tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=True)
        outputs = self.llm.generate(prompts=inputs, sampling_params=self.sampling_params)
        # res_vec.append(outputs[0].outputs[0].text)
        print(outputs[0].outputs[0].text)
        return outputs[0].outputs[0].text