# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:44:49 2019

@author: u.hyeyeon
    Ref : https://github.com/lovit/textrank/

Description : TextRank based Summarizer (Keyword and key-sentence extractor)
"""
import nltk
import pandas as pd
from textrank import KeywordSummarizer
from textrank import KeysentenceSummarizer

def komoran_tokenizer(sent):
    words = nltk.pos_tag(nltk.word_tokenize(sent))
    return words

path = "E:/u.hyeyeon/hyeyeon.github.git/TextRank/results/"
csvf = open(path+"rankSentence.tsv", 'w', encoding='utf-8')
ordf = open(path+"orderSentence.tsv", 'w', encoding='utf-8')
summarizer = KeysentenceSummarizer(tokenize = komoran_tokenizer, min_sim = 0.3)    

df = pd.read_csv('ROCStories_winter2017.csv', sep='\t',  header=0)


for i in df.index :
    storyid = df.iloc[i]["storyid"]
    storytitle = df.iloc[i]["storytitle"]
    sents = [df.iloc[i]["sentence1"],df.iloc[i]["sentence2"],df.iloc[i]["sentence3"],df.iloc[i]["sentence4"],df.iloc[i]["sentence5"]]
    print(i , "=============", storyid, ":", storytitle)
    
    keysents = summarizer.summarize(sents, topk=5)
    
    # rank 계산을 못했을 경우 (중복되는 단어가 없을 경우, 너무 적을 경우(1) 오류 발생)
    if keysents is not None :
        order = []
        for rank in keysents :
            csvf.write(storyid + "\t" + str(rank[0]+1) + "\t" + str(rank[1]) + "\t" + rank[2] + "\n")
            order.append(str(rank[0]+1))
            
        ordf.write(storyid + "\t" + "\t".join(order) + "\n")
    
        csvf.flush()
        ordf.flush()


csvf.close()
ordf.close()