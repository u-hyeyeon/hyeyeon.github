# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:44:49 2019

@author: u.hyeyeon
    Ref : https://github.com/lovit/textrank/

Description : TextRank based Summarizer (Keyword and key-sentence extractor)
"""
import pandas as pd
from textrank import KeywordSummarizer

df = pd.read_csv('ROCStories_winter2017.csv', sep='\t',  header=0)

keyword_extractor = KeywordSummarizer(
    tokenize = lambda x:x.split(),      # YOUR TOKENIZER
    window = -1,
    verbose = False
)

keywords = keyword_extractor.summarize(print(r[2:]), topk=30)
for word, rank in keywords:
    print(word, rank)