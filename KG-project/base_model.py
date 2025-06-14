import re
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
import pandas as pd
import os
import glob
import re

max_model_len=8192
tp_size=1
model_name='THUDM/glm-4-9b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
llm =LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    enforce_eager=True,
    # gpu_memory_utilization=0.9
)
stop_token_ids=[151329,151336,151338]
sampling_params=SamplingParams(
    temperature=1,               ### 生成温度，控制生成文本的多样性
    max_tokens=1024,                ### 生成的最大标记数
    stop_token_ids=stop_token_ids   ### 停止生成的标记ID列表
)

directory='KG_db/csv'
filename_list=glob.glob(os.path.join(directory,'*.csv'))

for filename in filename_list:
    input_vec=[]
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            input_vec.append(line)
    with open('new_prompt/prompt_init', 'r', encoding='utf-8') as f:
        prompt_init = f.read()

    with open('new_prompt/prompt_examples', 'r', encoding='utf-8') as f:
        prompt_examples = f.read()

    with open('new_prompt/prompy_start', 'r', encoding='utf-8') as f:
        prompt_start = f.read()

    res_vec=[]

    for input_txt in input_vec:
        prompt = prompt_init + '\n' + prompt_examples + '\n' + prompt_start + "### " + input_txt + " ###"
        # 应用聊天模板到提示词，生成输入
        prompt_input = [{'role': 'user', 'content': prompt}]

        inputs = tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=True)
        # 使用vLLM模型生成文本
        outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

        print(outputs[0].outputs[0].text)
        res_vec.append(outputs[0].outputs[0].text)
    out_csv = []
    for triple in res_vec:
        triple_tep = re.findall(r"Triple\(Subject\('(.*?)'\), Rel\('(.*?)'\), Object\('(.*?)'\)\)", triple)
        out_csv.extend(triple_tep)

    outfilename=filename.replace('.csv','_triple.csv')
    df = pd.DataFrame(out_csv, columns=['Subject', 'Rel', 'Object'])
    df.to_csv(outfilename, index=False, encoding='utf-8')

    print(f'{outfilename} has been saved')



