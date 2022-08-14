from time import process_time_ns
from potion import Request, NotionHeader
from potion.api import *
from potion.objects import *
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import fasttext.util

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
import requests
import io
import PyPDF2
from PyPDF2 import PdfFileReader

nltk.download('punkt')
nltk.download('stopwords')


# Extract keywords with Yake
kw_extractor = yake.KeywordExtractor()


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


for res in result['results']:
    url = res['properties']['URL']['url']
    autotagged = res['properties']['Autotagged']['checkbox']
    if autotagged is True:
        continue
    # 
    print("################################",autotagged, "-----", url, "########################################")
    if url.endswith('pdf'):
        print("We have a pdf")
        r = requests.get(url)
        f = io.BytesIO(r.content)
        reader = PdfFileReader(f)
        text = reader.getPage(0).extractText()
    else:
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
        except:
            print("Error retrieving the page")

    if not text:
        continue

    ic(text)

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
    
    ic(scoring.mmr(doc_embedding, candidate_embeddings, candidates, 5, 5))
    ic(scoring.max_sum_similarity(doc_embedding, candidate_embeddings, candidates, 5, 5))

    print("Processing with YAKE")
    from yake import KeywordExtractor as Yake
    yake = Yake(lan="en")
    yake_keyphrases = yake.extract_keywords(text)

    ic(yake_keyphrases)

    print("Text rank ")
    import spacy
    retained_keywords = []
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank", config={ "stopwords": { "word": ["NOUN"] } })
    doc = nlp(text)
    for phrase in doc._.phrases[:5]:
        ic(phrase)
        retained_keywords.append(phrase.text)
    print("================== ")

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("topicrank")
    doc = nlp(text)
    for phrase in doc._.phrases[:5]:
        ic(phrase)

    existing_tags = ic(res['properties']['Tags']['multi_select'])

    new_tags = []
    for name in retained_keywords:
        new_tags.append(prop.MultiSelectOption(name=name))
    for old_tag in existing_tags:
        ic(old_tag)
        new_tags.append(prop.MultiSelectOption(name=old_tag['name']))

    page_prop_data = Page(properties=prop.MultiSelect('Tags', new_tags))
    req.patch(url=page_update(res['id']), data=page_prop_data)

    page_prop_data = Page(properties=Properties(prop.CheckBox('Autotagged', True)))
    req.patch(url=page_update(res['id']), data=page_prop_data)
