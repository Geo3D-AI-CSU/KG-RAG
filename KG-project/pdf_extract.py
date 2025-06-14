import logging
import spacy
import pandas as pd
import pdfplumber
from pdf_parsing.src.parser import PDFParser

class ExtractPDF:
    def __init__(self,filename):
        self.filename = filename                    ### pdf的名称与目录
        self.nlp = spacy.load('zh_core_web_lg')     ### 针对pdf文本，调用nlp模型处理文本以分割句子

    ### 对初分割的句子进行整理
    def sentence_slove(self,sentences):
        res_sentences = []
        target=True
        tep=''
        for sentence in sentences:
            ###修改原始句子中的换行符等符号
            new_sentence=sentence.replace('\n', '').replace('\r', '').replace(' ', '')
            # 空句子处理
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
                res_sentences.append(new_sentence)

        return res_sentences

    ### pdf_to_sentences 顾名思义抽取pdf文档中的文本，并分割句子
    def pdf_to_sentences(self):

        # 首先利用pdfplumber获取文本内容
        with pdfplumber.open(self.filename) as pdf:
            # 对每一页的pdf文本进行处理
            pdf_sentences=[]
            for page in pdf.pages:
                # print(page.page_number)
                text = page.extract_text()
                text_nlp =self.nlp(text)
                text_sentences=[sent.text for sent in text_nlp.sents]
                for sentence in text_sentences:
                    pdf_sentences.append(sentence)
                # pdf_sentences.append(text_sentences)
        res_sentences=self.sentence_slove(pdf_sentences)
        DF_sentences=pd.DataFrame(res_sentences,columns=['sentence'])

        ### step2
        res_stop_sentences=[]
        res_sentence= ''
        for sentence_stop in DF_sentences['sentence']:
            doc_stop=self.nlp(sentence_stop)
            for word in doc_stop:
                if word.is_stop == False:
                    res_sentence = res_sentence + word.text
            res_stop_sentences.append(res_sentence)
            res_sentence= ''
        res_DF=pd.DataFrame(res_stop_sentences,columns=['sentence'])
        outfilename=self.filename.split('.')[0]+'_sentences.csv'
        res_DF.to_csv(outfilename,index=False)

        return print('句子分割完成!!!，生成csv文件名：'+outfilename)

    # 对pdf中的表格进行提取
    def pdf_to_tables(self):
        filenamelist=[]
        target_num=0
        with pdfplumber.open(self.filename) as pdf:
            for page in pdf.pages:
                table=page.extract_tables()
                for row in table:
                    df =pd.DataFrame(row[1:],columns=row[0])
                    if df.empty == False and df.isna().all().all() == False and (df != '').any().any()== True:
                        outfilename=self.filename.split('.')[0]+'_table_'+str(target_num)+'.csv'
                        filenamelist.append(outfilename)
                        df.to_csv(outfilename,index=False)
                        target_num=target_num+1

        print('表格分割完成!!!，生成csv文件列表：')
        return print(filenamelist)

    def pdf_to_images(self):
        parser=PDFParser(self.filename)
        parser.extract_images()
        target_num=0
        filelist=[]
        for image in parser.images:
            image_filename=self.filename.split('.')[0]+'_figure_'+str(target_num)+'.png'
            with open(image_filename,'wb') as writer:
                writer.write(image.image_data)
            target_num=target_num+1
            filelist.append(image_filename)
        print('图片分割完成!!!，生成figure文件列表')
        return print(filelist)












