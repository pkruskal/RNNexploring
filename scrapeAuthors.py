__author__ = 'peter'

import pandas as pd
import nltk
import numpy as np
import re
import gutenbergMetadata as gbMeta
import gutenbergText as gbTxt
import itertools

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


def gatherAuthorTexts(authorMetadata):

    authorMetadata = authorMetadata[authorMetadata['type']=='Text']
    authorMetadata = authorMetadata[authorMetadata['english']==True]

    textList = []
    for ind in authorMetadata.index:
        print('extractin ' + authorMetadata.ix[ind,'title'])
        text = gbTxt.load_etext(authorMetadata.ix[ind,'id'])
        text = gbTxt.strip_headers(text)
        textList.append(text)
        #text = re.sub('\\r',' \\r ',text)
        #text = re.sub('\\n',' \\n ',text)
        #text = re.sub('"',' " ',text)
        #textList.append(text.splitlines())

    return textList


def vocabularize(textList):
    #a bit of a catch all messy function
    #takes any list of lists of texts and will extract the vocabulary from it
    vocabList = []

    def loopTexts(textBranch):
        #trying to make things backword compatible for python 2.7 and 3
        try:
            #fails on python 3
            strFlag = type(textBranch[0]) == unicode
        except:
            strFlag = type(textBranch[0]) == str

        #recursivly check lists for their text compenents (this could get messy but does add flexability)
        if strFlag:
            #assumption here that if the first element is a str then they all are
            #tokenizedText = [nltk.word_tokenize(thisText) for thisText in textBranch]
            tokenizedText = [nltk.pos_tag(nltk.word_tokenize(thisText)) for thisText in textBranch]
            vocabList.extend(tokenizedText)
            print('finished sentences for a branch')
        else:
            for nextText in textBranch:
                loopTexts(nextText)

    #to keep things flexable this allows texts to be stored in lists of lists of arbitrary depth
    loopTexts(textList)

    #make a final array of words and do a freq dist on them
    word_freq = nltk.FreqDist(itertools.chain(*vocabList))

    #make this a pandas table
    wordStats = [(word[0][0],word[0][1],word[1]) for word in word_freq.items()]
    vocabDF = pd.DataFrame(wordStats,columns=['word','POS','wordCount'])
    vocabDF.sort('wordCount',ascending=False,inplace = True)

    #add in POS counts
    posCount = vocabDF.groupby('POS').count()
    posCount['posCount'] = posCount['wordCount']
    result = pd.merge(vocabDF, posCount, left_on='POS', right_index=True, how='left', sort=False)
    vocabDF['posCount'] = result['posCount']


    #truncate vocabulary based on minimum number of occurences of word or POS
    minOccurences = 3
    vocabDF['truncatedWords'] = vocabDF['word']
    #set words that don't occure often by their POS
    vocabDF['truncatedWords'][vocabDF['wordCount'] < minOccurences] = vocabDF['POS'][vocabDF['wordCount'] < minOccurences]
    #if even the POS doesn't occure often then set to a default place holder
    #ToDo :: I think there is a bug in this right now, need to make sure the count for POS is occuring just over the min occuring words
    vocabDF['truncatedWords'][vocabDF['wordCount'] < minOccurences][vocabDF['posCount'] < minOccurences] = 'PlaceHolder'
    #make truncated reference table
    truncatedVocab = vocabDF.drop_duplicates('truncatedWords')
    truncatedVocab['word'] = truncatedVocab['truncatedWords']
    del truncatedVocab['truncatedWords']

    #now look at weak VS robust words
    #minOccurences = 10
    #robustWords = vocabDF[vocabDF['count'] >= minOccurences]
    #weakWords = vocabDF[vocabDF['count'] < minOccurences]

    return truncatedVocab

    ###############

    # isolate POS for words below threshold
    #   check them here nltk.help.upenn_tagset()
    # count them and if below the threshold standardize them with a placeholder in the vocab DF
    # remove duplicates when the count is below a threshold and there are two with the same placeholder DF
    # create a DF for vocabTruncat
    # when a word is below threshold, replace the word with its POS
    # enumberate the vocabTruncate (or just reindex them?)
    # now can return the full vocabDF, and the vocabTruncate



    sortedVocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)



    robustWords = [x for x in sortedVocab if x[1] >= minOccurences]
    weakWords = [x for x in sortedVocab if x[1] < minOccurences]

    word2DiffPOS = [word for word in weakWords if word in robustWords]
    weakWordsPOS = np.unique([word[1] for word in weakWords])




    #now look for infrequent words
    #check if word exists with different parts of speach, and then match to that word
    #finnaly check if part of speach exists within the other low feq. occurring words




    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*vocabList))
    print("Found %d unique words tokens." % len(word_freq.items()))



    # need to get vocab
    # need to cut off vocab and then replace words with their part of speech
    return vocab

def sentenceTrainer(bookList):
    #isolate out all sentences
    sentences = [nltk.sent_tokenize(book) for book in bookList]
    return sentences

def chapterTrainer():
    #isolate chapters and then sentences

def paragraphTrainer():
    #try to isolate out paragraps?

def isolateTextCompentents(textList):
    '''
    take a list of texts and try to break them down
    :param textList: a list of strings presumed to be books
    :return sentenses: a list of sentences within each list of texts (books)
    :return chapters: a list of sentences within a list of chapters withing a list of texts (books)
    '''
    #

    chapterExpression = 'CHAPTER'

    for text in textList
    words = nltk.tokenize.wordpunct_tokenize(text)

    sentences = nltk.sent_tokenize(text)

    #could also use
    #Punkt Sentence Tokenizer
    #tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for
    # abbreviation words, collocations, and words that start sentences.
    #
    #sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #sentences2 = sent_detector.tokenize(text.strip())
    #
    #this seems to be identical however

    tokenizer = nltk.tokenize.RegexpTokenizer('CHAPTER',gaps=True)
    chapters = tokenizer.tokenize(text)

##### isolate Jane Austin Texts ####
def main():
    #TODO::
    # data currently stored in a parent directory but we should change this to the main directory
    # and then add a git ignore to keep it from synching
    metadata = pd.read_csv('../listOfEnglishDocs.csv')
    janeAustenData = metadata[metadata['author'] == 'Austen, Jane']
    #trying to just restrict to books
    janeAustenData = janeAustenData.ix[[101,115,133,150,153,908,1298]]

    janeAustenTexts = gatherAuthorTexts(janeAustenData)

    janeAustenSentences = sentenceTrainer(janeAustenTexts)

    vocab = vocabularize(janeAustenSentences)



powSentences = []
for isentence, sentence in enumerate(sentences):
    powSentences.append(nltk.pos_tag(nltk.tokenize.wordpunct_tokenize(sentence)))
    if np.mod(isentence,500) == 0:
        print('finished sentence ' + str(isentence) + ' out of ' + str(len(sentences)))
        print(powSentences[-1])

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
