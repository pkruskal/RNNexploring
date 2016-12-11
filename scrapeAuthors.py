__author__ = 'peter'

import pandas as pd
import nltk
import numpy as np
import re
import gutenbergMetadata as gbMeta
import gutenbergText as gbTxt
import itertools
import ast

def gatherMetadata():
    md = gbMeta.readmetadata()
    metadata = pd.DataFrame(md).T

    metadata['language'][metadata['language'] == None] = ['none']
    metadata['english'] = metadata['language'].map(lambda x : 'en' in x if x else False)
    englishMetadata = metadata[metadata['english'] == True]

    englishMetadata = englishMetadata[englishMetadata['type'] == 'Text']
    englishMetadata.to_csv('listOfEnglishDocs.csv',encoding='utf-8')


    #keywords = set()
    allWords = []
    for thisSet in englishMetadata['subjects']:
        #keywords = keywords.union(thisSet)
        allWords.extend(list(thisSet))
    from itertools import groupby
    a = groupby(allWords)
    keyWordCounts = []
    for key, group in a:
        numOccurences = len(list(group))
        if numOccurences > 1:
            keyWordCounts.append({
                'kewword' : key,
                'count' : numOccurences
            })
    keywordsPD = pd.DataFrame(keyWordCounts)
    keywordsPD.to_csv('keyword_counts.csv',encoding = 'utf-8')



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


def vocabularize(textList,minWordOccurance = 3):
    #a bit of a catch all messy function
    #takes any list of lists of texts and will extract the vocabulary from it
    vocabList = []
    tokenTextList = []
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
            tokenTextList.append(tokenizedText)
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

    vocabDF['truncatedWords'] = vocabDF['word']

    #count POS for minimally occuring words
    vocabDF['POSrestCount'] = vocabDF['posCount']
    posCountRestricted = vocabDF[vocabDF['wordCount'] < minWordOccurance].groupby('POS').count()
    result = pd.merge(vocabDF, posCountRestricted, left_on='POS', right_index=True, how='left', sort=False)
    vocabDF['POSrestCount'] = result['POSrestCount_y']
    #set words that don't occure often by their POS
    vocabDF['truncatedWords'][vocabDF['wordCount'] < minWordOccurance] = vocabDF['POS'][vocabDF['wordCount'] < minWordOccurance]

    #if even the POS doesn't occure often then set to a default place holder
    placeHolderIndex = vocabDF[vocabDF['wordCount'] < minWordOccurance][vocabDF['POSrestCount'] < minWordOccurance].index.values
    vocabDF.ix[placeHolderIndex,'truncatedWords'] = 'PlaceHolder'


    #make truncated reference table
    truncatedVocab = vocabDF.drop_duplicates('truncatedWords')
    truncatedVocab['word'] = truncatedVocab['truncatedWords']
    del truncatedVocab['truncatedWords']

    #now look at weak VS robust words
    #minOccurences = 10
    #robustWords = vocabDF[vocabDF['count'] >= minOccurences]
    #weakWords = vocabDF[vocabDF['count'] < minOccurences]

    return truncatedVocab

def oldVocabNotes():
    ###############

    # isolate POS for words below threshold
    #   check them here nltk.help.upenn_tagset()
    # count them and if below the threshold standardize them with a placeholder in the vocab DF
    # remove duplicates when the count is below a threshold and there are two with the same placeholder DF
    # create a DF for vocabTruncat
    # when a word is below threshold, replace the word with its POS
    # enumberate the vocabTruncate (or just reindex them?)
    # now can return the full vocabDF, and the vocabTruncate


    """
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
    """


def sentenceTrainer(bookList):
    #isolate out all sentences
    sentences = [nltk.sent_tokenize(book) for book in bookList]
    return sentences

def chapterTrainer():
    #isolate chapters and then sentences
    pass

def paragraphTrainer():
    #try to isolate out paragraps?
    pass

"""
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
"""

##### isolate Jane Austin Texts ####
def janeAusten():
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

    '''
    similar to jane auting
    While these works are not truly Jane in print, they are like Jane in their romantic spirit and are deserving of mention.</p>
        <br>
        <p><span class="textBold">Elizabeth Gaskell (1810-1865)</span> - "North and South", "Wives and Daughters"</p>
        <p><span class="textBold">Louisa May Alcott (1832-1888</span>) - "Little Women"</p>
        <p><span class="textBold">Wilkie Collins (1824-1889)</span> - "The Woman in White", "The Moonstone"</p>
        <p><span class="textBold">Anthony Trollope (1815-1882)</span> - "The Warden", "Barchester Towers"</p>
        <p><span class="textBold">Charles Dickens (1812-1870)</span> - "Little Dorrit", &quot;A Tale of Two Cities&quot;</p>
        <p><span class="textBold">Maria Edgeworth (1767-1849)</span> - "Belinda", "Castle Rackrent"</p>
        <p><span class="textBold">Frances Burney (1752-1840)</span> - "Camilla", "Cecilia", "Evelina"</p>
        <p><span class="textBold">Winston Graham (1908-2003)</span> - "Poldark"</p>

    '''

    return vocab, janeAustenSentences


def brit19thCentury():

    #key word look up
    keywords = ['English -- 19th century',]

    authors = ['Austen, Jane',]

    '''
    manual finds
    36621	Colman, George
    36515	Colman, George
    171	Rowson, Mrs.

    34238	Cobbett, William (not a novel)
    9365	Lamb, Mary (correspondence)
    10851	Lamb, Mary (correspondence)
    10130	Lamb, Mary (correspondence)

    19407	Morton, Thomas
    43529	Fenwick, E. (Eliza) (don't know if it's british)
    9321	Edgeworth, Maria

    8937	Edgeworth, Maria
    35638	Edgeworth, Maria
    9107	Edgeworth, Maria
    9439	Edgeworth, Maria
    8720	Edgeworth, Maria
    8826	Edgeworth, Maria
    9414	Edgeworth, Maria
    9620	Edgeworth, Maria
    11121	Edgeworth, Maria
    1424	Edgeworth, Maria
    8531	Edgeworth, Maria

    25895	Watts, Susannah	1768	1842	set([Leicester (England) -- Description and travel])

    49621	Opie, Amelia	1769	1853	set([Fathers and daughters -- Fiction, Psychological fiction, Seduction -- Fiction, Mentally ill -- Fiction, Domestic fiction, Young women -- Fiction])
    37908	Opie, Amelia	1769	1853	set([Feminist fiction, Domestic fiction, Mothers and daughters -- Fiction, Unmarried couples -- Fiction])
    35294	Opie, Amelia	1769	1853	set([Fiction])
    40180	Opie, Amelia	1769	1853	set([Fiction])

    35532	Wellington, Arthur Wellesley, Duke of	1769	1852	set([Wellington, Arthur Wellesley, Duke of, 1769-1852 -- Correspondence])

    8940	Foster, John	1770	1843	set([Great Britain -- Social conditions -- 19th century, Education, Religious education, Missions -- India, Literacy])

    Brown, Charles Brockden	1771	1810	set([Dementia -- Fiction, Murder -- Investigation -- Fiction, Psychological fiction, Epistolary fiction, Sleepwalking -- Fiction, Delaware Indians -- Fiction, Wilderness areas -- Fiction, Philadelphia (Pa.) -- History -- 18th century -- Fiction, Young men -- Fiction, Horror tales])
    Brown, Charles Brockden	1771	1810	set([Epistolary fiction])
    Brown, Charles Brockden	1771	1810	set([Murderers -- Fiction, Psychological fiction, Yellow fever -- Fiction, Philadelphia (Pa.) -- Fiction, Young men -- Fiction, Horror tales])
    Brown, Charles Brockden	1771	1810	set([Philadelphia (Pa.) -- History -- 19th century -- Fiction, Young women -- Fiction, Gothic fiction (Literary genre)])
    Brown, Charles Brockden	1771	1810	set([Philadelphia (Pa.) -- History -- 19th century -- Fiction, Young women -- Fiction, Gothic fiction (Literary genre)])
    Brown, Charles Brockden	1771	1810	set([Philadelphia (Pa.) -- History -- 19th century -- Fiction, Young women -- Fiction, Gothic fiction (Literary genre)])

    792	Brown, Charles Brockden	1771	1810	set([Psychological fiction, Epistolary fiction, Radicals -- Fiction, Gothic fiction (Literary genre), Fathers -- Death -- Fiction, Combustion, Spontaneous -- Fiction, Pennsylvania -- History -- Colonial period, ca. 1600-1775 -- Fiction, Horror tales, Brothers and sisters -- Fiction, Religious fanaticism -- Fiction])
    842	Brown, Charles Brockden	1771	1810	set([Ventriloquists -- Fiction])

    29725	Sherwood, Mrs. (Mary Martha)	1775	1851	set([Christian life -- Juvenile fiction])
    12315	Sherwood, Mrs. (Mary Martha)	1775	1851	set([Romanies -- England -- Juvenile fiction, Blacksmiths -- Juvenile fiction, Kindness -- Juvenile fiction, Abduction -- Juvenile fiction, Inheritance and succession -- Juvenile fiction, Christian life -- Juvenile fiction, Jews -- England -- Juvenile fiction, Diligence -- Juvenile fiction, Wealth -- Juvenile fiction])

    2049	Hazlitt, William	1778	1830	set([Authors, English -- 19th century -- Biography, Authors, English -- 19th century -- Correspondence, Hazlitt, William, 1778-1830 -- Relations with women, Hazlitt, William, 1778-1830 -- Correspondence, Love-letters, Imaginary letters])

    7948	Irving, Washington	1783	1859	set([Abbotsford (Scotland), Newstead Abbey])
    7994	Irving, Washington	1783	1859	set([American prose literature])
    7993	Irving, Washington	1783	1859	set([Authors, Irish -- 18th century -- Biography, Goldsmith, Oliver, 1730?-1774])
    1850	Irving, Washington	1783	1859	set([Christmas -- England, Christmas stories, American])
    20656	Irving, Washington	1783	1859	set([Christmas -- England, Christmas stories, American])
    8519	Irving, Washington	1783	1859	set([Columbus, Christopher, Explorers -- Spain -- Biography, America -- Discovery and exploration -- Spanish, Explorers -- America -- Biography])
    13515	Irving, Washington	1783	1859	set([England -- Social life and customs -- 19th century -- Fiction, National characteristics, English -- Fiction])
    14228	Irving, Washington	1783	1859	set([England -- Social life and customs -- 19th century -- Fiction, National characteristics, English -- Fiction])

    3480	Knowles, James Sheridan	1784	1862	set([English drama (Comedy)])
    3539	Knowles, James Sheridan	1784	1862	set([English drama, Comedies])

    2075	Peacock, Thomas Love	1785	1866	set([Authors -- Fiction, England -- Fiction, Philosophy -- Fiction])
    21514	Peacock, Thomas Love	1785	1866	set([English fiction -- 19th century])
    966	Peacock, Thomas Love	1785	1866	set([Historical fiction, Women outlaws -- Fiction, Great Britain -- History -- Richard I, 1189-1199 -- Fiction, Robin Hood (Legendary character) -- Fiction, Maid Marian (Legendary character) -- Fiction, Adventure stories, Sherwood Forest (England) -- Fiction, Love stories])
    9909	Peacock, Thomas Love	1785	1866	set([Humorous stories, Gothic fiction (Literary genre)])
    12803	Peacock, Thomas Love	1785	1866	set([Upper class -- England -- Fiction, Satire, English])
    7830	Vaux, Frances Bowyer	1785	1854	set([Children -- Conduct of life, Conduct of life -- Juvenile literature])


    22844	Mitford, Mary Russell	1787	1855	set([City and town life -- Fiction, Businesswomen -- Fiction, Single women -- Fiction, Short stories])
    22841	Mitford, Mary Russell	1787	1855	set([City and town life -- Fiction, Short stories])
    22839	Mitford, Mary Russell	1787	1855	set([England -- Social life and customs -- 19th century -- Fiction, Country life -- England -- Fiction, Pastoral fiction])
    22835	Mitford, Mary Russell	1787	1855	set([England -- Social life and customs -- 19th century -- Fiction, Pastoral fiction, Country life -- England -- Fiction, Short stories])
    22837	Mitford, Mary Russell	1787	1855	set([England -- Social life and customs -- 19th century -- Fiction, Pastoral fiction, Country life -- England -- Fiction, Short stories])
    22836	Mitford, Mary Russell	1787	1855	set([England -- Social life and customs -- 19th century -- Fiction, Short stories, Country life -- England -- Fiction, Pastoral fiction])
    22843	Mitford, Mary Russell	1787	1855	set([Inheritance and succession -- Fiction, Short stories])
    22845	Mitford, Mary Russell	1787	1855	set([Mate selection -- Fiction, Short stories, Young women -- Fiction])
    22838	Mitford, Mary Russell	1787	1855	set([Pastoral fiction, Courtship -- Fiction, Country life -- England -- Fiction, England -- Social life and customs -- 19th century -- Fiction, Widows -- Fiction, Short stories])
    22842	Mitford, Mary Russell	1787	1855	set([Poor families -- Fiction, Short stories, Dogs -- Fiction])
    22840	Mitford, Mary Russell	1787	1855	set([Short stories, Boarding school students -- Fiction])
    22846	Mitford, Mary Russell	1787	1855	set([Short stories, Poor children -- Fiction])
    2496	Mitford, Mary Russell	1787	1855	set([Villages -- England -- Fiction, Country life -- England -- Fiction, Pastoral fiction])

    44996	Scargill, William Pitt	1787	1836	set([])
    43756	Scargill, William Pitt	1787	1836	set([England -- Social life and customs -- 19th century -- Fiction])
    44159	Scargill, William Pitt	1787	1836	set([England -- Social life and customs -- 19th century -- Fiction])
    44959	Scargill, William Pitt	1787	1836	set([England -- Social life and customs -- 19th century -- Fiction])
    52375	Scargill, William Pitt	1787	1836	set([Epistolary fiction, Women -- Conduct of life -- Fiction, Women -- Education -- Fiction])
    40974	Scargill, William Pitt	1787	1836	set([Women -- Conduct of life -- Fiction, Epistolary fiction, Women -- Education -- Fiction])

    40158	Panache, Madame	1789	1881	set([England -- Social life and customs -- Fiction])
    40159	Panache, Madame	1789	1881	set([England -- Social life and customs -- Fiction])
    40160	Panache, Madame	1789	1881	set([England -- Social life and customs -- Fiction])

    21556	Marryat, Frederick	1792	1848	set([Adventure stories, Western stories, Travelers -- Fiction, Indians of North America -- Fiction, Historical fiction, French -- West (U.S.) -- Fiction])
    13673	Marryat, Frederick	1792	1848	set([Adventure stories])
    21571	Marryat, Frederick	1792	1848	set([Adventure stories])
    13276	Marryat, Frederick	1792	1848	set([Africa -- Fiction, Adventure stories])
    21555	Marryat, Frederick	1792	1848	set([Africa -- Fiction, Adventure stories])
    21549	Marryat, Frederick	1792	1848	set([Barges -- Fiction, Seafaring life -- Fiction, Great Britain. Royal Navy -- Officers -- Fiction, Orphans -- Fiction, Thames River (England) -- Fiction, Picaresque literature, Sea stories, Great Britain -- History, Naval -- 19th century -- Fiction])
    22496	Marryat, Frederick	1792	1848	set([Canada -- Fiction, Pioneers -- Fiction])
    24211	Marryat, Frederick	1792	1848	set([Canada -- Fiction, Pioneers -- Fiction])
    31579	Marryat, Frederick	1792	1848	set([Drama, Dialogues, Short stories, Fiction])
    15991	Marryat, Frederick	1792	1848	set([England -- Fiction, Foundlings -- Fiction])
    24470	Marryat, Frederick	1792	1848	set([England -- Fiction, Foundlings -- Fiction])
    21574	Marryat, Frederick	1792	1848	set([England -- Social life and customs -- 19th century -- Fiction, Murder -- Fiction, Fathers and sons -- Fiction, Poachers -- Fiction])
    23952	Marryat, Frederick	1792	1848	set([Fiction])
    12954	Marryat, Frederick	1792	1848	set([Flying Dutchman -- Fiction, Immortalism -- Fiction, Sea stories, Horror tales])
    21573	Marryat, Frederick	1792	1848	set([Flying Dutchman -- Fiction, Immortalism -- Fiction, Sea stories, Horror tales])
    13010	Marryat, Frederick	1792	1848	set([Great Britain. Royal Navy -- Officers -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction, Sea stories])
    21554	Marryat, Frederick	1792	1848	set([Great Britain. Royal Navy -- Officers -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction, Sea stories])
    13405	Marryat, Frederick	1792	1848	set([Historical fiction, Western stories, Travelers -- Fiction, Indians of North America -- Fiction, Adventure stories, French -- West (U.S.) -- Fiction])
    21577	Marryat, Frederick	1792	1848	set([Midshipmen -- Fiction])
    13148	Marryat, Frederick	1792	1848	set([Napoleonic Wars, 1800-1815 -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction, Sea stories, English, Smugglers -- Fiction, Midshipmen -- Fiction])
    21572	Marryat, Frederick	1792	1848	set([Napoleonic Wars, 1800-1815 -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction])
    6471	Marryat, Frederick	1792	1848	set([New Forest (England : Forest) -- Fiction, Orphans -- Fiction, Great Britain -- History -- Civil War, 1642-1649 -- Fiction])
    21558	Marryat, Frederick	1792	1848	set([New Forest (England : Forest) -- Fiction, Orphans -- Fiction, Great Britain -- History -- Civil War, 1642-1649 -- Fiction])
    21550	Marryat, Frederick	1792	1848	set([Nore Mutiny, 1797 -- Fiction, Great Britain -- History, Naval -- 18th century -- Fiction])
    21580	Marryat, Frederick	1792	1848	set([Pirates -- Fiction])
    14222	Marryat, Frederick	1792	1848	set([Poor -- Fiction, Sea stories])
    21575	Marryat, Frederick	1792	1848	set([Poor -- Fiction, Sea stories])
    25719	Marryat, Frederick	1792	1848	set([Privateering -- Fiction, Historical fiction, Sea stories])
    12959	Marryat, Frederick	1792	1848	set([Romantic suspense fiction, Sea stories, Historical fiction, Merchant mariners -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction, East India Company -- Fiction])
    21557	Marryat, Frederick	1792	1848	set([Romantic suspense fiction, Sea stories, Historical fiction, Merchant mariners -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction, East India Company -- Fiction])
    21576	Marryat, Frederick	1792	1848	set([Sea stories, Historical fiction, Privateering -- Fiction])
    12558	Marryat, Frederick	1792	1848	set([Seafaring life -- Fiction, Dogs -- Fiction, Historical fiction, Great Britain. Royal Navy -- Fiction])
    21579	Marryat, Frederick	1792	1848	set([Seafaring life -- Fiction, Dogs -- Fiction, Historical fiction, Great Britain. Royal Navy -- Fiction])
    29291	Marryat, Frederick	1792	1848	set([Seafaring life -- Fiction, Pirates -- Fiction, Smugglers -- Fiction, Sea stories, Great Britain. Royal Navy -- Fiction])
    21552	Marryat, Frederick	1792	1848	set([Shipwreck survival -- Fiction, Shipwrecks -- Fiction, Adventure stories, Islands -- Fiction])
    6897	Marryat, Frederick	1792	1848	set([Shipwreck survival -- Juvenile fiction, Missionaries -- Juvenile fiction, Islands of the Pacific -- Juvenile fiction])
    21551	Marryat, Frederick	1792	1848	set([Shipwreck survival -- Juvenile fiction, Missionaries -- Juvenile fiction, Islands of the Pacific -- Juvenile fiction])
    1412	Marryat, Frederick	1792	1848	set([Shipwrecks -- Fiction, Survival -- Fiction, Islands -- Fiction, Sea stories, Adventure stories])
    21559	Marryat, Frederick	1792	1848	set([Smugglers -- Fiction, Sea stories])
    23137	Marryat, Frederick	1792	1848	set([United States -- Social life and customs -- 1783-1865, Canada -- Description and travel, Marryat, Frederick, 1792-1848 -- Travel -- America, United States -- Description and travel])
    23138	Marryat, Frederick	1792	1848	set([United States -- Social life and customs -- 1783-1865, Canada -- Description and travel, Marryat, Frederick, 1792-1848 -- Travel -- America, United States -- Description and travel])
    23139	Marryat, Frederick	1792	1848	set([Voyages and travels])
    6629	Marryat, Frederick	1792	1848	set([War stories, Napoleonic Wars, 1800-1815 -- Fiction, Autobiographical fiction, Sea stories, Adventure stories, Midshipmen -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction, Young men -- Fiction])
    21553	Marryat, Frederick	1792	1848	set([War stories, Napoleonic Wars, 1800-1815 -- Fiction, Autobiographical fiction, Sea stories, Adventure stories, Midshipmen -- Fiction, Great Britain -- History, Naval -- 19th century -- Fiction, Young men -- Fiction])


    40405	Grey, Mrs. (Elizabeth Caroline)	1798	1869	set([Great Britain -- Social life and customs -- 19th century -- Fiction, Mate selection -- Fiction, Young women -- Fiction])
    40406	Grey, Mrs. (Elizabeth Caroline)	1798	1869	set([Great Britain -- Social life and customs -- 19th century -- Fiction, Mate selection -- Fiction, Young women -- Fiction])
    40407	Grey, Mrs. (Elizabeth Caroline)	1798	1869	set([Great Britain -- Social life and customs -- 19th century -- Fiction, Mate selection -- Fiction, Young women -- Fiction])

    28784	Borrow, George	1803	1881	set([Authors, English -- 19th century -- Correspondence, Borrow, George, 1803-1881 -- Correspondence])
    28814	Borrow, George	1803	1881	set([Authors, English -- 19th century -- Correspondence, Borrow, George, 1803-1881 -- Correspondence])

    452	Borrow, George	1803	1881	set([England -- Fiction, Romanies -- Fiction])
    18400	Borrow, George	1803	1881	set([England -- Fiction, Romanies -- Fiction])
    20198	Borrow, George	1803	1881	set([England -- Fiction, Romanies -- Fiction])
    23287	Borrow, George	1803	1881	set([England -- Fiction, Romanies -- Fiction])
    30792	Borrow, George	1803	1881	set([England -- Fiction, Romanies -- Fiction])

    2733	Borrow, George	1803	1881	set([Romanies -- England, Romani poetry, Romanies -- Languages])
    25071	Borrow, George	1803	1881	set([Romanies -- Fiction, Adventure stories, England -- Fiction])
    422	Borrow, George	1803	1881	set([Romanies -- Fiction, England -- Fiction])
    21206	Borrow, George	1803	1881	set([Romanies -- Fiction, England -- Fiction])
    22877	Borrow, George	1803	1881	set([Romanies -- Fiction, England -- Fiction])
    22878	Borrow, George	1803	1881	set([Romanies -- Fiction, England -- Fiction])

    7685	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Crime -- Fiction, English fiction -- 19th century, Detective and mystery stories, English, London (England) -- Fiction])
    7686	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Crime -- Fiction, English fiction -- 19th century, Detective and mystery stories, English, London (England) -- Fiction])
    7687	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Crime -- Fiction, English fiction -- 19th century, Detective and mystery stories, English, London (England) -- Fiction])
    7688	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Crime -- Fiction, English fiction -- 19th century, Detective and mystery stories, English, London (England) -- Fiction])
    7689	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Crime -- Fiction, English fiction -- 19th century, Detective and mystery stories, English, London (England) -- Fiction])
    7690	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Crime -- Fiction, English fiction -- 19th century, Detective and mystery stories, English, London (England) -- Fiction])
    7691	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Crime -- Fiction, English fiction -- 19th century, Detective and mystery stories, English, London (England) -- Fiction])
    7750	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([England -- Social life and customs -- 19th century -- Fiction])
    7751	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([England -- Social life and customs -- 19th century -- Fiction])
    7752	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([England -- Social life and customs -- 19th century -- Fiction])
    7753	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([England -- Social life and customs -- 19th century -- Fiction])
    7754	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([England -- Social life and customs -- 19th century -- Fiction])
    7755	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([England -- Social life and customs -- 19th century -- Fiction])
    7756	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([England -- Social life and customs -- 19th century -- Fiction])

    7586	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7587	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7588	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7589	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7590	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7591	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7592	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7593	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7594	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7595	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7596	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7597	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7598	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7599	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7601	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7602	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7603	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7604	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])
    7605	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century, Families -- Fiction])

    2461	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7631	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7632	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7633	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7634	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7635	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7636	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7637	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7638	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7639	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7640	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7641	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7642	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7643	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7644	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7645	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7646	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7647	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7648	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7649	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7650	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7651	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7652	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7653	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7654	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7655	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7656	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7657	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7658	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7659	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7660	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7661	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7662	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7663	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7664	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7665	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7666	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7667	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7668	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7669	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7670	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7671	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7692	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7693	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7694	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7695	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7696	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7697	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7698	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7699	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    7701	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9763	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9764	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9765	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9766	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9767	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9768	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9769	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9770	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9771	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9772	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9773	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9774	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction -- 19th century])
    9750	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction])
    9751	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction])
    9752	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction])
    9753	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction])
    9754	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction])
    9755	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([English fiction])

    7757	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Fiction])
    7758	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Fiction])
    7759	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Fiction])
    7760	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Fiction])
    7761	Lytton, Edward Bulwer Lytton, Baron	1803	1873	set([Fiction])

    7926	Disraeli, Benjamin, Earl of Beaconsfield	1804	1881	set([England -- Social life and customs -- 19th century -- Fiction])	14	{u'text/html; charset=utf-8': u'http://www.gutenberg.org/files/7926/7926-h/7926-h.htm', u'text/plain; charset=us-ascii': u'http://www.gutenberg.org/files/7926/7926.txt', u'text/plain; charset=utf-8': u'http://www.gutenberg.org/files/7926/7926-0.txt', u'application/zip': u'http://www.gutenberg.org/files/7926/7926.zip', u'application/rdf+xml': u'http://www.gutenberg.org/ebooks/7926.rdf', u'application/epub+zip': u'http://www.gutenberg.org/ebooks/7926.epub.images', u'application/x-mobipocket-ebook': u'http://www.gutenberg.org/ebooks/7926.kindle.noimages'}	7926	set([PR])	[en]	Endymion	Text	TRUE
    7842	Disraeli, Benjamin, Earl of Beaconsfield	1804	1881	set([Fiction])
    20008	Disraeli, Benjamin, Earl of Beaconsfield	1804	1881	set([Nobility -- Fiction, Political fiction, Bildungsromans, Fathers and sons -- Fiction, Catholic emancipation -- Fiction, Young men -- Fiction, Love stories])
    9840	Disraeli, Benjamin, Earl of Beaconsfield	1804	1881	set([Political fiction, British -- Europe -- Fiction, Politicians -- Fiction, Bildungsromans, Young men -- Fiction])
    7412	Disraeli, Benjamin, Earl of Beaconsfield	1804	1881	set([Political fiction, Politicians -- Fiction, Great Britain -- Fiction])


    '''






"""


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

"""