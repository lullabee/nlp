import spacy
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from yake import KeywordExtractor as Yake
from icecream import ic

from lib import scoring


def extractKeyPhrasesWithBert(text):    
    print("Processing with Bert")
    # Extract candidate words/phrases
    vectorizer = KeyphraseCountVectorizer()
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, vectorizer=vectorizer,  use_mmr=True)
    return keywords

def extractKeywordsWithYake(text, top_n):
    print("Processing with YAKE")
    yake = Yake(lan="en")
    yake_keyphrases = yake.extract_keywords(text)
    return yake_keyphrases
    

def extractKeywordsWithBert(text, top_n):
    n_gram_range = (1, 2)
    stop_words = "english"

    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    candidates = count.get_feature_names_out()
    # ic(candidates)

    model = SentenceTransformer('all-mpnet-base-v2')
    # model = SentenceTransformer('stsb-roberta-large')

    #('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)
    # ic(candidate_embeddings.shape)
    # ic(doc_embedding.shape, candidate_embeddings.shape)

    # ic(scoring.mmr(doc_embedding, candidate_embeddings, candidates, top_n, 10))

    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    
    # ic(scoring.mmr(doc_embedding, candidate_embeddings, candidates, top_n, 1))
    # ic(scoring.max_sum_similarity(doc_embedding, candidate_embeddings, candidates, top_n, 20))

def extractCommonWords(text):
    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            print ('Type: ', nltk_result.label(), 'Name: ', name)


def extractKeywordsWithSpacyTextRank(text, top_n):
    print("Text rank ")
    retained_keywords = []
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank", config={ "stopwords": { "word": ["NOUN"] } })
    doc = nlp(text)
    for phrase in doc._.phrases[:top_n]:
        # ic(phrase)
        retained_keywords.append(phrase.text)
    return retained_keywords

def extractKeywordsWithSpacyTopicRank(text, top_n):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("topicrank")
    doc = nlp(text)
    retained_keywords = []

    for phrase in doc._.phrases[:top_n]:
        # ic(phrase)
        retained_keywords.append(phrase.text)
    return retained_keywords

def dedup(candidates1, candidates2, threshold):
    model = KeyBERT()
    embedding1 = model.encode(candidates1)
    embedding2 = model.encode(candidates2)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    ic(candidates1, candidates2)
    ic("Similarity score:", cosine_scores)
