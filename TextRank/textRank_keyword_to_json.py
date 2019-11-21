# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:44:49 2019

@author: u.hyeyeon
    Ref : https://github.com/lovit/textrank/

Description : TextRank based Summarizer (Keyword and key-sentence extractor)
"""
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add(".")


import pandas as pd
from textrank import KeysentenceSummarizer

keyword_extractor = KeysentenceSummarizer(
    tokenize = lambda x:x,
    #tokenize = lambda x:_tokenize(x),
    min_sim = 0.5,
    verbose = True
)
"""
def _tokenize(sent):
    #words = komoran.pos(sent, join=True)
    words = nltk.pos_tag(nltk.word_tokenize(sent))
    words = [w for w in words if ('NN' in w or 'XR' in w or 'VA' in w or 'VV' in w)]
    print(words)
    return words
"""

df = pd.read_csv('ROCStories_winter2017.csv', sep='\t',  header=0)
data = []
for i in df.index :
    storyid = df.iloc[i]["storyid"]
    docs = [df.iloc[i]["sentence1"],df.iloc[i]["sentence2"],df.iloc[i]["sentence3"],df.iloc[i]["sentence4"],df.iloc[i]["sentence5"]]
    #wordsList = nltk.word_tokenize(df.iloc[i]["sentence1"])    # 1 sentence
    wordsList = nltk.word_tokenize(" ".join(docs))              # 5 sentence
    wordsList = [w for w in wordsList if not w in stop_words]
    
    # TextRank based keyword extraction
    keysents = keyword_extractor.summarize(wordsList, topk=10)

    print(df.iloc[i]["storyid"])
    print(keysents)   # (rank, word)

    data.append({'storyid' : df.iloc[i]["storyid"],
        'storytitle':df.iloc[i]["storytitle"],
        'sentences' : docs,
        'calcSimSents' : keyword_extractor.R.shape[0],
        'rank' :keysents })    
    
df2 = pd.DataFrame(data)
df2.to_json(path + "/df_keyword.json") 
    
"""
keywords = keyword_extractor.summarize(print(r[2:]), topk=30)
for word, rank in keywords:
    print(word, rank)
"""