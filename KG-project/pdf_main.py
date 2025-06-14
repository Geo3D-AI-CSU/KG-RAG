#该项目旨在负责文档最开始的数据清洗
import spacy
import pandas as pd
import pdfplumber
from pdf_parsing.src.parser import PDFParser
import re
import os
import glob

# 指定要搜索的目录
directory = 'KG_db/pdf'
# 获取目录下所有 PDF 文件名的完整路径
filename_list = glob.glob(os.path.join(directory, '*.pdf'))

nlp=spacy.load('zh_core_web_lg')
keywords=[""]

for filename in filename_list:
    print(f'-----------------Processing {filename}')
    with (pdfplumber.open(filename)as pdf):
        pdf_sentences = []
        res_sentences = []
        print('step1---------------------------')
        for page in pdf.pages:
            text=page.extract_text()
            text_nlp=nlp(text)
            text_sentences=[sent.text for sent in text_nlp.sents]
            for sentence in text_sentences:
                pdf_sentences.append(sentence)
        print('step2***************************')
        tep= ' '
        for sentence in pdf_sentences:
            new_sentence = sentence.replace('\n', '').replace('\r', '').replace(' ', '')
            if new_sentence == '':
                continue
            if new_sentence.isdigit():
                continue
            if new_sentence.count('.') >= 5:
                continue

            if (new_sentence[-1] == '.') or (new_sentence[-1] == '）') or (new_sentence[-1] == '：') or (
                    new_sentence[-1] == ')') or (new_sentence[-1] == '～') or (re.search(r'\d$', new_sentence)) or (
                    new_sentence[-1] == "/"
            ) or (new_sentence[-1] == '—') or (re.search(r'[a-zA-Z]$', new_sentence)) or (
                    re.search(r'[\u4e00-\u9fa5]$', sentence)) or (new_sentence[-1] == '、') or (new_sentence[-1]=='，'):
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
            # print(new_sentence)



    res_sentences = pd.DataFrame(res_sentences)
    # outfilename=filename.replace('.pdf','.csv')
    outfilename=filename.replace('pdf','csv')
    res_sentences.to_csv(outfilename,index=False,header=False)
    print(f'---------------CSV保存完毕: {outfilename}')
