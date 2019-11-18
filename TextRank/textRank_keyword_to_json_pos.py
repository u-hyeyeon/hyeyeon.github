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
stop_words.add(",")

import pandas as pd
#from textrank import KeywordSummarizer
from textrank import KeysentenceSummarizer
from collections import OrderedDict

keyword_extractor = KeysentenceSummarizer(
    tokenize = lambda x:x,
    min_count = 0,
    min_sim = 0.3,
    verbose = True
)

def _tokenize(sent):
    #words = komoran.pos(sent, join=True)
    words = nltk.pos_tag(nltk.word_tokenize(sent))
    words = [w[0] for w in words if ('NN' in w or 'XR' in w or 'VA' in w or 'VV' in w)]
    
    return words

path = "E:/u.hyeyeon/hyeyeon.github.git/TextRank/results/"
df = pd.read_csv('ROCStories_winter2017.csv', sep='\t',  header=0)
data = []

for i in df.index :
    docs = [df.iloc[i]["sentence1"],df.iloc[i]["sentence2"],df.iloc[i]["sentence3"],df.iloc[i]["sentence4"],df.iloc[i]["sentence5"]]
    
    wordsList = _tokenize(" ".join(docs))              # 5 sentence
    wordsList = [w for w in wordsList if not w in stop_words]
    if not wordsList :
        continue
    
    # TextRank based keyword extraction
    keysents = keyword_extractor.summarize(wordsList, topk=30)

    print(df.iloc[i]["storyid"])
    print(keysents)   # (rank, word)

    data.append({'storyid' : df.iloc[i]["storyid"],
        'storytitle':df.iloc[i]["storytitle"],
        'sentences' : docs,
        'calcSimSents' : keyword_extractor.R.shape[0],
        'rank' :keysents })
    # =========================================================================

df2 = pd.DataFrame(data)
df2.to_json(path + "/df_keyword_pos.json") 
"""
df = pd.read_json(path+"/df_keyword_apply_tokenize.json")
"""