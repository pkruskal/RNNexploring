__author__ = 'peter'

import pandas as pd
import nltk
import numpy as np
import re
import gutenbergMetadata as gbMeta
import gutenbergText as gbTxt


def gatherMetadata():
    md = gbMeta.readmetadata()
    metadata = pd.DataFrame(md).T

    metadata['language'][metadata['language'] == None] = ['none']
    metadata['english'] = metadata['language'].map(lambda x : 'en' in x if x else False)
    englishMetadata = metadata[metadata['english'] == True]

    englishMetadata = englishMetadata[englishMetadata['type'] == 'Text']
    englishMetadata.to_csv('listOfEnglishDocs.csv',encoding='utf-8')

    authors = englishMetadata.groupby('author').count()
    authors['count'] = authors['authoryearofbirth']
    authors = authors[['count','authoryearofbirth','authoryearofdeath']]
    authors = authors.sort('count',ascending = False)
    authors.reset_index(inplace=True)
    authors['authoryearofbirth'] = authors['author'].map(lambda x: englishMetadata['authoryearofbirth'][englishMetadata['author'] == x].values[0])
    authors['authoryearofdeath'] = authors['author'].map(lambda x: englishMetadata['authoryearofdeath'][englishMetadata['author'] == x].values[0])
    authors.to_csv('listOfAuthors.csv',encoding='utf-8')

    return metadata, authors


def gautherAuthorTexts(author,metadata):
    author = 'Austen, Jane'
    metadata = pd.read_csv('listOfEnglishDocs.csv')

    authorData = metadata[metadata['author'] == author]
    authorData = authorData[authorData['type']=='Text']
    authorData = authorData[authorData['english']==True]

    textList = []
    for ind in authorData.index:
        text = gbTxt.load_etext(ind)
        text = gbTxt.strip_headers(text)
        text = re.sub('\\r',' \\r ',text)
        text = re.sub('\\n',' \\n ',text)
        #text = re.sub('"',' " ',text)
        textList.append(text.splitlines())


lines = text.splitlines()
for i,line in enumerate(lines):
    if len(lines[i]) == 0:
        lines[i] = '[  newParagraph  ]'
text2 = ' '.join(lines)
paragraphs = re.split('\[  newParagraph  \]',text2)



#getting lines of text from a book via it's text number
text = gbTxt.load_etext(textNum)
text = gbTxt.strip_headers(text)
lines = text.splitlines()



import urllib
import re
import time
import sys



authorListUrl = 'https://www.gutenberg.org/browse/authors/a'

response = urllib.urlopen(authorListUrl)
html = response.read()
authorsText = re.split('class="pgdbbyauthor"',html)[1]
authorList = re.split('<h2>',authorsText)

for author in authorList:
    authorData = {}
    checkAuthorInfo = re.findall('name=(.*?)a>',author)
    print checkAuthorInfo
    if len(checkAuthorInfo) > 0:
        checkAuthorInfo = re.findall('">(.*?)<',checkAuthorInfo[0])[0]
        #chck if dates over the 1000s exist
        dates = re.findall('\d{4}',checkAuthorInfo)
        if len(dates) == 2:
            authorData['dateStart'] = dates[0]
            authorData['dateEnd'] = dates[1]

_GUTENBERG_CATALOG_URL = \
    r'http://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2'

from rdflib import plugin
from rdflib.graph import Graph
from rdflib.store import Store
from rdflib.term import URIRef

