import pandas as pd
from fuzzywuzzy import process

class ContextExpander:
    def __init__(self, filepath, query, target_length=300):
        """
        初始化 ContextExpander
        :param filepath: CSV 文件路径
        :param query: 模糊匹配的关键词
        :param target_length: 扩展段落的目标字符长度
        """
        self.filepath = filepath
        self.query = query
        self.target_length = target_length
        self.sentences = []
        self.context_paragraphs = []

    def load_data(self):
        """加载 CSV 文件中的句子"""
        try:
            df = pd.read_csv(self.filepath, header=None, names=["句子"])
            self.sentences = df["句子"].tolist()
            print(f"成功加载 {len(self.sentences)} 条句子")
        except Exception as e:
            print(f"加载数据时出错: {e}")

    def get_context_by_length(self, index):
        """
        根据目标字符串长度向上和向下扩展句子
        :param index: 匹配句子的索引
        :return: 扩展后的段落（列表形式）
        """
        context = [self.sentences[index]]  # 包含当前句子
        total_length = len(self.sentences[index])  # 当前段落的总长度

        # 向上扩展两句
        up_index = index - 1
        up_context = []
        for _ in range(2):
            if up_index >= 0:
                up_context.insert(0, self.sentences[up_index])
                total_length += len(self.sentences[up_index])
                up_index -= 1

        # 向下扩展直到目标长度
        down_index = index + 1
        down_context = []
        while total_length < self.target_length and down_index < len(self.sentences):
            down_context.append(self.sentences[down_index])
            total_length += len(self.sentences[down_index])
            down_index += 1

        return up_context + context + down_context

    def expand_contexts(self, limit=10):
        """
        执行模糊匹配并扩展上下文段落
        :param limit: 模糊匹配返回的句子数量
        """
        if not self.sentences:
            print("请先加载数据！")
            return

        # 模糊匹配关键词
        matched_sentences = process.extract(self.query, self.sentences, limit=limit)

        # 获取上下文段落
        self.context_paragraphs = []
        for match in matched_sentences:
            matched_sentence = match[0]
            index = self.sentences.index(matched_sentence)
            context = self.get_context_by_length(index)
            self.context_paragraphs.append(context)

    def result_ans(self):
        """打印扩展后的段落"""
        if not self.context_paragraphs:
            print("未找到扩展段落，请先调用 expand_contexts 方法！")
            return
        result=''
        for i, context in enumerate(self.context_paragraphs):
            result += f"\n段落 {i + 1}:\n{''.join(context)}"
        return result