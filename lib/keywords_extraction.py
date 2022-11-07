import nltk
import spacy
import yake
from icecream import ic
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk import ne_chunk, pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tree import Tree
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from yake import KeywordExtractor as Yake
import umap
import hdbscan
import matplotlib.pyplot as plt
from bertopic import BERTopic
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF



from lib import scoring

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Extract keywords with Yake
kw_extractor = yake.KeywordExtractor()

def extractThemes(files_data):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(files_data, show_progress_bar=True)
    umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='cosine').fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)
    result = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    

def extractKeyPhrasesWithBert(text, tags, top_n, threshold):    
    print("Processing with Bert")
    final_keywords = []
    # Extract candidate words/phrases
    vectorizer = KeyphraseCountVectorizer()
    kw_model = KeyBERT(model='multi-qa-MiniLM-L6-cos-v1')
    # all-mpnet-base-v2
    keywords = kw_model.extract_keywords(text, seed_keywords=list(tags), vectorizer=vectorizer, top_n=top_n, use_maxsum=True)
    for keyword in keywords:
        if keyword[1] > threshold:
            final_keywords.append(keyword[0])
    ic(final_keywords)
    return final_keywords

def extractKeywordsWithYake(text, top_n):
    print("Processing with YAKE")
    yake = Yake(lan="en", top = top_n)
    yake_keyphrases = yake.extract_keywords(text)
    return yake_keyphrases
    
def extractKeywordsWithBert(text, top_n):
    n_gram_range = (1, 2)
    count = CountVectorizer(ngram_range=n_gram_range, stop_words="english").fit([text])
    candidates = count.get_feature_names_out()

    model = SentenceTransformer('all-mpnet-base-v2')
    # model = SentenceTransformer('stsb-roberta-large')

    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)

    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    return keywords

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

def extractTopicsWithBert(text, top_n):
    # try:
    model = BERTopic()
    topics, probabilities = model.fit_transform(text)
    ic(model.get_topic_freq().head())
    ic(model.get_topics(top_n))
    model.visualize_topics()
    # except:
        # print("Error happened")

def extractKeywordsWithSpacyTopicRank(text, top_n):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("topicrank")
    doc = ie_preprocess(nlp(text))
    retained_keywords = []

    for phrase in doc._.phrases[:top_n]:
        # ic(phrase)
        retained_keywords.append(phrase.text)
    return retained_keywords

def polyFuzzy(candidates1, candidates2):
    tfidf = TFIDF(n_gram_range=(3, 3), min_similarity=0, model_id="TF-IDF-Sklearn")
    model = PolyFuzz(tfidf)
    # model = PolyFuzz("TF-IDF")

    final_candidates = []
    to_remove = []

    model.match(candidates1, candidates2)
    model.visualize_precision_recall(kde=True)

    candidates = model.get_matches()
    # for i in range(len(candidates)):
        # if candidates["Similarity"][i] > 0.7:
        # final_candidates.append(min(candidates["From"][i], candidates["To"][i], key=len))


    ic(candidates)
    final_candidates = (list(set(final_candidates)))
    return final_candidates

def dedup(candidates1, candidates2, threshold):
    model = SentenceTransformer('all-mpnet-base-v2')
    embedding1 = model.encode(candidates1)
    embedding2 = model.encode(candidates2)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    final_candidates = []
    for i in range(len(candidates1)):
        for j in range(len(candidates2)):
            if cosine_scores[i][j].item() < threshold:
                final_candidates.append(candidates1[i])
                final_candidates.append(candidates2[j])
                # ic(cosine_scores[i][j].item(), candidates1[i], candidates2[j])
            else:
                final_candidates.append(min([candidates1[i], candidates2[j]], key=len))
    final_candidates = (list(set(final_candidates)))
    return final_candidates

def ie_preprocess(document):
    stop = stopwords.words('english')
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences
