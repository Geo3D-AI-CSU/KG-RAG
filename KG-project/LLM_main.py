from dev_model import dev_model
from prompt_cot import property_prompt
from fuzzkeword import ContextExpander

#全局信息
csv_file=''
csv_file_alter=''
filename=''

#提示词信息
subject='XX'
rel='XX'
object='XX'
obejct_example='XX'
example='[]'

query_list=['常德-安仁断裂','老屋湾——朝北塘断层','小汾源——单家湾断裂','常德-安仁北西向基底断裂','连鱼塘——牯牛岭断层','石牛塘——庙湾断层','青山冲——肖家湾断层','花桥——蒋家新屋断层']



if __name__=='__main__':

    dev_model = dev_model()
    ans_list=[]
    for query_tep in query_list:
        query=query_tep+object

        rag_model=ContextExpander(csv_file_alter,query,target_length=400)
        rag_model.load_data()
        rag_model.expand_contexts(limit=10)

        prompt_model=property_prompt()
        prompt_model.subject_init(query_tep)
        prompt_model.rel_init(rel)
        prompt_model.object_init(object)
        prompt_model.example_init(example)

        init_rag=rag_model.result_ans()
        print(init_rag)

        init_dev=prompt_model.prompt_dev_1(init_rag)
        dev_ans_1=dev_model.llm_txt_generate(init_dev)

        init_ans=prompt_model.prompt_dev_2(init_rag,dev_ans_1)
        dev_ans_2=dev_model.llm_txt_generate(init_ans)

        info_dev_3=prompt_model.prompt_dev_3(dev_ans_2)
        ans_dev_3=dev_model.llm_txt_generate(info_dev_3)



        info_result=prompt_model.prompt_guide(ans_dev_3)
        result=dev_model.llm_txt_generate(info_result)

        info_dev_4=prompt_model.prompt_dev_4(result)
        print(info_dev_4)
        ans_dev_4=dev_model.llm_txt_generate(info_dev_4)

        info_result_alter=prompt_model.prompt_ans(ans_dev_4)
        result_alter=dev_model.llm_txt_generate(info_result_alter)

        ans_list.append(result_alter)

    print("ans------------------------------------")
    for ans in ans_list:
        print(ans)

