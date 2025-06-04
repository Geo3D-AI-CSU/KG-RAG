# 暂时是batch模型的封装
from typing import Union
from transformers import AutoTokenizer,LogitsProcessorList,AutoModelForCausalLM


class batch_model:
    def __init__(self,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',trust_remote_code=True).eval()

    def process_model_outputs(self,inputs,outputs):
        responses = []
        for inputs_ids, outputs_ids in zip(inputs.input_ids,outputs):
            response = self.tokenizer.decode(outputs_ids[len(inputs_ids):], skip_special_tokens=True).strip()
            responses.append(response)

        return responses

    def batch(self,
              messages:Union[str,list[str]],
              max_input_tokens:int=8192,
              max_new_tokens:int=8192,
              num_beams:int=1,
              do_sample:bool=True,
              top_p:float=0.8,
              temperature:float=0.8,
              logits_processor=None):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        messages = [messages] if isinstance(messages,str) else messages
        batched_inputs=self.tokenizer(messages,
                                      return_tensors='pt',
                                      padding="max_length",
                                      truncation=True,max_length=max_input_tokens).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            "eos_token_id":self.model.config.eos_token_id
        }

        batched_outputs = self.model.generate(**batched_inputs,**gen_kwargs)
        batched_response = self.process_model_outputs(batched_inputs,batched_outputs)
        return batched_response

    def generate(self,batch_message):
        batch_inputs = []
        max_input_tokens = 1024
        for i, messages in enumerate(batch_message):
            # print(messages)
            new_batch_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            max_input_tokens = max(max_input_tokens, len(new_batch_input))
            # print(new_batch_input)
            batch_inputs.append(new_batch_input)
        gen_kwargs = {
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": 8192,
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.8,
            "num_beams": 1,
        }
        batch_response = self.batch(batch_inputs, **gen_kwargs)
        for response in batch_response:
            print("=" * 10)
            print(response)

        return batch_response