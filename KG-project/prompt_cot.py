import csv
import re

class property_prompt:
    def __init__(self):
        self.subject=''
        self.object=''
        self.rel=''
        self.example=''

    def subject_init(self, input):
        self.subject = input

    def object_init(self, input):
        self.object = input

    def rel_init(self, input):
        self.rel = input

    def example_init(self, input):
        self.example = input

    def prompt_init(self):
        info_object = f'请告诉我有关于{self.subject}的相关信息'
        return  info_object

    def prompt_introduction(self):
        result=f'假设您是一位地质学专家，现在请你对{self.object}概念做出简短解释。'
        return result

    def prompt_object(self,input_subject,input):
        result=f'假设您是一位地质学专家，现在需要你帮助完成实体搜索匹配与对齐工作。首先请你根据提供的知识内容中总结符合**{self.object}**概念的实体，{self.object}概念常见的实体表达如{input}。现在具体的知识内容如下：**{input_subject}。需要注意的是，搜索到的实体需要确保与{self.subject}存在上下文关联。'
        return result

    def prompt_rule(self,input,rule):
        result=f'假设您是一位地质学专家，现在需要你帮助完成实体概念是否正确的判断.当前分析的实体概念是**{self.object}**,我们认为符合判断概念的例子如：[{rule}],不符合判断的概念如：[岩浆期后气—液蚀变,多种蚀变]。请你根据知识内容对所有的实体进行判断，并最终输出符合正确判断的实体列表。知识内容如下：{input}'
        return result

    def prompt_rel(self,input):
        result=f'假设您是一位地质学专家，现在需要你帮助完成实体关系三元组生成工作。请你根据提供的知识内容中**{self.object}**概念对应的实体信息同所有{self.subject}是否匹配关系{self.rel}。该关系生成三元组的常见表达如:{self.example}。现在具体的知识内容如下：**{input}**。请在回答中只保留正确匹配**{self.rel}**关系信息，形成[{self.subject},{self.rel},{self.object}对应的概念实体]结构的知识三元组，同时要求三元组[Subject, Rel ,Object]内容符合语义顺序。'
        return result



    def prompt_dev_1(self,input):
        result=f'您的任务是将给定文本中符合关键词{self.subject}和{self.rel}内容的段落进行筛选，筛选的目的是确定{self.subject}的{self.object}内容，并对该内容进行全面分析，逐点分析，切记不要遗漏。给定文本内容如下：{input}'
        print('*' * 10, 'dev_1')
        return result

    def prompt_dev_2(self,input,dev_input):
        # result=f'假设您是一位地质学专家，现在需要你帮助完成实体关系三元组生成工作。请你根据提供的知识内容{dev_input}以及原文段落{input}总结结构为{self.subject}和{self.rel}所构成的三元组。三元组示例为{self.example},回答要求为三元组列表。'
        result = f'假设您是一位地质学专家，现在需要你帮助完成实体关系三元组生成工作。请你根据提供的知识内容进行知识抽取工作。要求只抽取由实体**{self.subject}**和关系**{self.rel}**所组成的知识。知识最终以三元组形式进行回答，当前要求关系{self.rel}的三元组示例为{self.example},请参考示例格式并回答三元组列表，切记不要表达其余关系。对生成的每一个三元组进行判断，如果三元组不满足事实和关系要求，请删除该三元组。给定知识内容如下：{dev_input}'
        print('*' * 10, 'dev_2')
        return result

    def prompt_dev_3(self,input):
        result = f'假设您是一位地质学专家,请你对提供的三元组内容进行判断，逐一分析每个三元组有关于{self.rel}的表达是否合理和满足要求。要求三元组表达为[subject,rel,object]。其中subject表示断裂构造的名称，rel表示断裂构造走向这一关系，object表示断裂构造的具体方向值，首先subject的概念层级需要匹配验证；其次rel的关系概念不能被替换，必须匹配；最后object要求为具体走向方向，严禁表达为方向字符之外的其他属性。如果没有相应结果，配以无法提供。对不能严格匹配与合理的三元组进行删除。提供三元组如下：{input}。'
        print('*' * 10, 'dev_3')
        return result

    def prompt_guide(self,input):
        result=f'您的任务是将给定文本描述中被描述为符合匹配关系且判断合理的知识转换为三元组列表形式的语义图，对于不匹配和不合理的内容不进行抽取。切记三元组的形式必须是[Subject, Rel ,Object]。在答案中，请严格只包含三元组列表，不要包含任何解释或道歉。回答要求格式请严格遵守如下示例，不要有多余的字符和格式；[Subject_1, Rel_1 , Object_1],[Subject_2, Rel_2, Object_2],[Subject_3, Rel_3, Object_3]。给定文本如下：{input}'
        return result

    def prompt_dev_4(self,input):
        result = (f'假设您是一位地质学专家,请你对提供的三元组知识内容进行规范化。回答的三元组要求如下：对于断裂构造长度相关三元组的表达应分为[subject,rel,object]三部分。例如{self.example}。'
                  f'三元组的实体允许用——字符进行连接，如果提供的三元组中的object项表达为无法提供，请在回答中删除该三元组。现在请你根据输入知识进行有效规范化，输入内容如下：{input}')
        print('*' * 10, 'dev_4')
        return result

    def prompt_ans(self,input):
        result=f'您的任务是将给定三元组进行结构化表达，不要对提供内容的三元组作拆分。切记三元组的形式必须是[Subject, Rel ,Object]，三元组内部的实体无需引号备注。如果提供文本符合匹配的结果只存在一个三元组则直接输出，如果存在多个三元组则采用三元组列表表达，列表表达要求格严格遵守如下示例，不要有多余的字符。三元组列表格式；[Subject_1, Rel_1 , Object_1],[Subject_2, Rel_2, Object_2],[Subject_3, Rel_3, Object_3]。在答案中，请严格只包含三元组或三元组列表，不要包含任何解释或道歉。给定文本如下：{input}'
        print('-'*20)
        return result

    def triple_csv(self, filename, data):
        pattern = r"[\[（]\s*(.*?)\s*[，,]\s*(.*?)\s*[，,]\s*(.*?)\s*[\]）]"
        triples = re.findall(pattern, data)
        print(f"找到的三元组: {triples}")

        # 读取现有 CSV 文件中的三元组
        existing_triples = set()
        try:
            with open(filename, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)  # 跳过表头
                for row in reader:
                    # 将现有三元组存入集合，使用元组作为集合的元素（去重）
                    existing_triples.add(tuple(row))
        except FileNotFoundError:
            # 如果文件不存在，说明这是首次创建文件，跳过读取步骤
            existing_triples = set()

        # 将新的三元组过滤，只添加不存在的三元组
        new_triples = [triple for triple in triples if tuple(triple) not in existing_triples]

        if new_triples:
            # 追加新的三元组到文件
            with open(filename, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # 如果文件为空，先写入表头
                    writer.writerow(["subject", "rel", "object"])
                writer.writerows(new_triples)  # 写入新三元组
            print(f"新增三元组已追加至 {filename}")
        else:
            print("没有新的三元组需要添加。")