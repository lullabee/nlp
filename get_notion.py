from time import process_time_ns
from potion import Request, NotionHeader
from potion.api import *
from potion.objects import *
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

from gensim.summarization import summarize
import trafilatura
import scoring
import graphviz
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import random as rd
import yake
import nltk   
import pytextrank
from icecream import ic

nltk.download('punkt')
nltk.download('stopwords')

# Notion only
token = 'secret_SLpSNU22LjOaAgCVYVDMxlpJvYlyX6PsPxwgX9fAUyv'
database_id = '9ff7e448072143c89d437ad965eb92d7'
# database_id = '7fc2e672f2aa4204a871798df6d6fcd7'
# database_id = '12da1b182aba475da17e6402d1e17d99'
database_id = 'aae63f1339094e31b4af7c37591e8ada'

nh = NotionHeader(authorization=token)
req = Request(nh.headers)
 
data = Filter.QueryDatabase(query.QueryProperty(property_name='URL',
                            property_type=FilterProperty.rich_text,
                            conditions=FilterCondition.Text.is_not_empty,
                            condition_value=True))

result = req.post(url=database_query(database_id=database_id), data=data)

# Extract keywords
kw_extractor = yake.KeywordExtractor()

dot = graphviz.Digraph(comment='The Round Table')   

for res in result['results']:
    url = res['properties']['URL']['url']

    print("################################", url, "########################################")
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        print(text)
    except:
        print("Error retrieving the page")
        continue
    
    if not text:
        continue
    n_gram_range = (1, 2)
    stop_words = "english"

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    candidates = count.get_feature_names()
    # print(candidates)

    print("Processing with DNN")
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)

    # top_n = 5
    # distances = cosine_similarity(doc_embedding, candidate_embeddings)
    # keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    # print(keywords)
    
    ic(scoring.mmr(doc_embedding, candidate_embeddings, candidates, 5, 10))
    ic(scoring.max_sum_similarity(doc_embedding, candidate_embeddings, candidates, 5, 10))

    print("Processing with YAKE")
    from yake import KeywordExtractor as Yake
    yake = Yake(lan="en")
    yake_keyphrases = yake.extract_keywords(text)

    ic(yake_keyphrases)

    print("Text rank ")
    import spacy

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank", config={ "stopwords": { "word": ["NOUN"] } })
    doc = nlp(text)
    for phrase in doc._.phrases[:10]:
        ic(phrase)
    print("================== ")

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("topicrank")
    doc = nlp(text)
    for phrase in doc._.phrases[:10]:
        ic(phrase)

    # import topicrank

    # # created a directed graph
    # graph=nx.gnp_random_graph(25,0.6,directed=True)
    # #draw a graph
    # nx.draw(graph,with_labels=True,font_color='red',font_size=10,node_color='yellow')
    # #plot a graph
    # plt.show()
    
    # for j in range(len(text)):
    #     tr = topicrank.TopicRank(text[j])
    #     print("Keywords of article", str(j+1), "\n", tr.get_top_n(n=5, extract_strategy='first'))


