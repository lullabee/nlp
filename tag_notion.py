import io
import requests
import trafilatura
from icecream import ic
from PyPDF2 import PdfFileReader

from lib.keywords_extraction import dedup, extractCommonWords, extractKeyPhrasesWithBert, extractKeywordsWithYake, extractTopicsWithBert, extractThemes, polyFuzzy

# Sensing 
# database_id = 'aae63f1339094e31b4af7c37591e8ada'

# ??????
# database_id = '0f6ccd8556194d409fab1fd427db3155'

# Weblinks
database_id = '9ff7e448072143c89d437ad965eb92d7'

# Companies and organisations
# database_id = '4fa6ddb87da547a488fa81f0fbafa21f' 

# Sensors
# database_id = '8209b1a3ceaa4b17ad83df0b253ff150'

from lib.notion import getLinksFromNotionDatabase, notionExtractDatabaseTags, notionSetAutoKeywords, notionSetAutotag


# notionExtractDatabaseTags(database_id)

top_n = 10
texts = []

ic("Getting links and tags from Notion")
notion_list = getLinksFromNotionDatabase(database_id)
tags = notionExtractDatabaseTags(database_id)

for notion_item in notion_list['results']:
    url = notion_item['properties']['URL']['url']
    autotagged = notion_item['properties']['Autotagged']['checkbox']
    if autotagged is True:
        continue
    
    print("################################", autotagged, "-----", url, "########################################")
    print("0. Fetching url")

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
            print("Error retrieving the page. At this point we can only process weblinks with the url field set")
            continue

    if not text:
        print("Error retrieving the page. At this point we can only process weblinks with the url field set")
        continue

    print("1. Computation started")
    keywords = extractKeyPhrasesWithBert(text, tags, top_n, 0.2)
    # keywords = [item[0] for item in keywords]
    

    # Yake +++++++++
    yake_keywords = extractKeywordsWithYake(text, top_n)
    yake_keywords = [item[0] for item in yake_keywords]
    ic(yake_keywords)
    # keywords = dedup(keywords, yake_keywords, 0.5)
    print("2. Deduplication")

    matches = polyFuzzy(keywords, yake_keywords)
    ic(matches)

    keywords = dedup(keywords, keywords, 0.5)
    ic(keywords)
    # common_words = extractCommonWords(text)

    # ic(keywords, yake_keywor ds_name)

    notionSetAutoKeywords(notion_item, keywords)
    # notionSetAutotag(notion_item['id'])
    texts.append(text)

# extractThemes(texts)