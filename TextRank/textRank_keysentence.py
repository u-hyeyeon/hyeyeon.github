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
    i = 5372
    storyid = df.iloc[i]["storyid"]
    storytitle = df.iloc[i]["storytitle"]
    sents = [df.iloc[i]["sentence1"],df.iloc[i]["sentence2"],df.iloc[i]["sentence3"],df.iloc[i]["sentence4"],df.iloc[i]["sentence5"]]
    keysents = summarizer.summarize(sents, topk=5)
    #print(keysents, "\n")
    print(i,":",storytitle)
    
    order = []
    for rank in keysents :
        csvf.write(storyid + "\t" + str(rank[0]+1) + "\t" + str(rank[1]) + "\t" + rank[2] + "\n")
        order.append(str(rank[0]+1))
        
    ordf.write(storyid + "\t" + "\t".join(order) + "\n")
    
    csvf.flush()
    ordf.flush()
    break

csvf.close()
ordf.close()