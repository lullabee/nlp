import io
from unittest import result

import nltk
import requests
import trafilatura
import yake
from icecream import ic
from PyPDF2 import PdfFileReader

from lib.keywords_extraction import extractCommonWords, extractKeyPhrasesWithBert, extractKeywordsWithYake


# database_id = 'aae63f1339094e31b4af7c37591e8ada'
database_id = '9ff7e448072143c89d437ad965eb92d7'
database_id = '0f6ccd8556194d409fab1fd427db3155'


from lib.notion import getLinksFromNotionDatabase, notionSetAutoKeywords, notionSetAutotag, notionExtractDatabaseTags

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Extract keywords with Yake
kw_extractor = yake.KeywordExtractor()

tags = notionExtractDatabaseTags(database_id)

top_n = 10

notion_list = getLinksFromNotionDatabase(database_id)

for notion_item in notion_list['results']:
    url = notion_item['properties']['URL']['url']
    autotagged = notion_item['properties']['Autotagged']['checkbox']
    # if autotagged is True:
    #     continue
    
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
            # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            # text = tokenizer.tokenize(data)

        except:
            print("Error retrieving the page. At this point we can only process weblinks with the url field set")
            continue

    if not text:
        print("Error retrieving the page. At this point we can only process weblinks with the url field set")
        continue


    keywords = extractKeyPhrasesWithBert(text, tags, top_n, 0.4)
    keywords_name = [item[0] for item in keywords]
    
    yake_keywords = extractKeywordsWithYake(text, top_n)
    yake_keywords_name = [item[0] for item in yake_keywords]

    common_words = extractCommonWords(text)
    # page_prop_data = Page(properties=prop.MultiSelect('AutoTags', new_tags))
    # req.patch(url=page_update(res['id']), data=page_prop_data)

    ic(keywords_name, yake_keywords_name)

    notionSetAutoKeywords(notion_item, keywords_name)
    notionSetAutotag(notion_item['id'])
