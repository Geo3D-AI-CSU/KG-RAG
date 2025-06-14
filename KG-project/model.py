from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class Model:

    def __init__(self,config):
        self.model=config['model_name']
        self.tensor_parallel_size=config['tp_size']
        self.max_model_len=config['max_model_len']
        self.trust_remote_code=config['trust_remote_code']
        self.enforce_eager=config['enforce_eager']
        self.temperature=config['temperature']
        self.max_tokens=config['max_tokens']
        self.stop_token_ids=config['stop_token_ids']

    def generate_text(self,prompt):

        tokenizer =AutoTokenizer.from_pretrained(self.model,trust_remote_code=self.trust_remote_code)

        llm=LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            enforce_eager=self.enforce_eager
        )

        sampling_params=SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids
        )

        res_list=[]

        number=1

        for idx_p in prompt:
            prompt = [{'role': 'user', 'content': idx_p}]

            inputs=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)

            outputs=llm.generate(prompts=inputs,sampling_params=sampling_params)

            res_list.append(outputs[0].outputs[0].text)

            print("[",number,"]","--",outputs[0].outputs[0].text)

            number+=1
        return res_list



