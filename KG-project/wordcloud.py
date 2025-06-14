#该项目旨在对构成的原初知识库进行词云演示
import spacy
import wordcloud
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

filename='output/湖南省衡南县毛湾矿区钨矿普查总结报告_res.csv'
DF=pd.read_csv(filename)
# nlp = spacy.load('zh_core_web_lg')

text = ' '.join(DF.iloc[:, [0, 2]].astype(str).agg(' '.join, axis=1))
print(text)

words = text.split()
word_freq = Counter(words)

# 删除前 N 个最常见的词
N = 20  # 例如，删除前 5 个最常见的词
most_common = word_freq.most_common(N)

# 从字典中删除这些词
for word, _ in most_common:
    del word_freq[word]

wc = wordcloud.WordCloud(font_path='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                         width=1000,
                         height=700,
                         background_color='white',
                         max_words=50,
                         )

wc.generate_from_frequencies(word_freq)


wc.to_file('wc.png')
plt.imshow(wc)
plt.axis('off')
plt.show()