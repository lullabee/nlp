from potion import NotionHeader, Request
from potion.api import *
from potion.objects import *
from icecream import ic

token = 'secret_SLpSNU22LjOaAgCVYVDMxlpJvYlyX6PsPxwgX9fAUyv'
nh = NotionHeader(authorization=token)
req = Request(nh.headers)

def getLinksFromNotionDatabase(database_id):
    
    data = Filter.QueryDatabase(query.QueryProperty(property_name='URL',
                                property_type=FilterProperty.rich_text,
                                conditions=FilterCondition.Text.is_not_empty,
                                condition_value=True))

    result = req.post(url=database_query(database_id=database_id), data=data)
    return result



def notionSetAutotag(id):
    page_prop_data = Page(properties=Properties(prop.CheckBox('Autotagged', True)))
    req.patch(url=page_update(id), data=page_prop_data)

def notionSetAutoKeywords(notion_item, keywords):
    # existing_tags = notion_item['properties']['AutoTags']['multi_select']
    new_tags = []
    for name in keywords:
        new_tags.append(prop.MultiSelectOption(name=name))
    # for old_tag in existing_tags:
    #     new_tags.append(prop.MultiSelectOption(name=old_tag['name']))
    # ic(new_tags)
    page_prop_data = Page(properties=prop.MultiSelect('AutoTags', new_tags))
    try:
        req.patch(url=page_update(notion_item['id']), data=page_prop_data)
    except:
        ic("C'est la merde")

