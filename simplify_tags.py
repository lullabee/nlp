from icecream import ic
from PyPDF2 import PdfFileReader
# from lib.keywords_extraction import dedup, extractCommonWords, extractKeyPhrasesWithBert, extractKeywordsWithYake, extractTopicsWithBert, extractThemes, polyFuzzy

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

top_n = 8
texts = []

ic("Getting links and tags from Notion")
tags = notionExtractDatabaseTags(database_id)
autotags = notionExtractDatabaseTags(database_id, 'AutoTags')
ic(tags)
ic(autotags)