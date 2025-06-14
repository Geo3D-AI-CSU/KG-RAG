import spacy
import pandas as pd
from langchain.document_loaders import UnstructuredPDFLoader

class PDF_sentence_csv:
    def __init__(self, filename_list):
        self.filename_list = filename_list
        self.nlp=spacy.load('zh_core_web_lg')

    def pdf_sentence_csv_generator(self):
        for filename in self.filename_list:
            loader = UnstructuredPDFLoader(filename)
            data = loader.load()

            #第一阶段，抽取句子
            res_step_1=[]
            for sentence_step_1 in data:
                sentence_step_1_txt=sentence_step_1.page_content
                sentence_step_1_nlp=self.nlp(sentence_step_1_txt)
                sentences_step_1=[sent.text for sent in sentence_step_1_nlp.sents]
                target=True
                tep= ''
                for sentence in sentences_step_1:
                    new_sentence=sentence.replace('\n', '').replace('\r', '').replace(' ', '')
                    #空句子处理
                    if new_sentence == '':
                        continue

                    # 句子如果是数字，针对pdf页码处理
                    if new_sentence.isdigit():
                        continue

                    # 去除一些符号的错误分割，保持句子完整性，改变target以确保句子多合一
                    if (new_sentence[-1] == '.') or (new_sentence[-1] == '）') or (new_sentence[-1] == '：') or (
                            new_sentence[-1] == ')') or (new_sentence[-1] == '～'):
                        # print(new_sentence)
                        tep = tep + new_sentence
                        target = False
                    else:
                        target = True
                        new_sentence = (tep + new_sentence)
                    if target == False:
                        continue
                    if target == True:
                        tep = ''
                        res_step_1.append(new_sentence)
                    # print('修改：', new_sentence)

             #第一阶段收尾，数据格式整理
            DF_step_1=pd.DataFrame(res_step_1,columns=['sentence'])

            #第二阶段，停止词去除
            res_step_end=[]
            res_sentence= ''
            for sentence_stop in DF_step_1['sentence']:
                doc_stop=self.nlp(sentence_stop)
                for word in doc_stop:
                    if word.is_stop == False:
                        res_sentence = res_sentence + word.text
                res_step_end.append(res_sentence)
                res_sentence = ''
            res_DF=pd.DataFrame(res_step_end,columns=['sentence'])
            outfilename=filename.split('.')[0]+'.csv'
            res_DF.to_csv(outfilename,index=False)

            #显示成功与否
            print('句子分割完成!!!，生成csv文件名：'+outfilename)

        print('所有文档分割完成，拟导入langchain。')