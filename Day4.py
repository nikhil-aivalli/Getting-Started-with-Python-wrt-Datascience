import nltk
nltk.download()

from nltk.tokenize import sent_tokenize,word_tokenize
text ="The government shut down at 12:01  on Saturday, after senators blocked a bill that would have kept the federal government running another few weeks  extended a childrens health insurance program. The legislation came up wa vote of 50-49 – about 10 votes short of passing. Democrats largely opposed the bill because it did include legal protections  DREAMers, the undocumented immigrants who came to the U.were covered under an Obama-era order. "
#sentences from the text created above
print (sent_tokenize(text))

#words from the text v=created above
print(word_tokenize(text))

from nltk.corpus import stopwords
stop=set(stopwords.words('English'))

stop=list(stop)
stop.append(',')
stop.append('.')

w=word_tokenize(text)
#####
clean_text=[i for i in w if not i in stop] #list comprehenstion
######
#OR
#####
clean_text=[]
for i in w:
    if i not in stop:
        clean_text.append(i)
#####
print(stop)
print(w)
print(clean_text)

################
stop=set(stopwords.words('French'))

########################

#Stemming

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize
ps=PorterStemmer()
#stems from words
ps.stem('updating')

for i in text:
    print(ps.stem(i))
    
#lemmatizing
#  similar operation to stemming
    
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

print(lemmatizer.lemmatizlemmatize("cats"))
print(lemmatizer.lemmatizlemmatize("cacti"))
print(lemmatizer.lemmatizlemmatize("geese"))
print(lemmatizer.lemmatizlemmatize("rocks"))
print(lemmatizer.lemmatizlemmatize("python"))
print(lemmatizer.lemmatizlemmatize("better",pos="a")) #pos-part of speech


####################################################
#wordnet is a nltk dictionary

from nltk.corpus import wordnet
#finding synonym
syns=wordnet.synsets("good")
print(syns)

print(syns[0].name())
print(syns[0].definition())
print(syns[3].definition())
print(syns[3].examples())

#creating the antonyms and synonym using lemmas

synonyms=[]
antonyms=[]

for syn in wordnet.synsets("pretty"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            
print("synonyms for preety are:",set(synonyms))            
print("Antonyms for preety are:",set(antonyms))

###########################################################

file=open("C:/Users/nikhi/Desktop/soft skills",'r')
file.readline()
file.readlines()

import requests
f_online='http://www.gutenberg.org/files/11111/11111.txt'
f_rawtext=requests.get(f_online).text
print(f_rawtext)

raw_words = word_tokenize(f_rawtext[:100])

tags=nltk.pos_tag(raw_words)
print(tags[:9])

from nltk.corpus import gutenberg
textguten=gutenberg.raw("bible-kjv.txt")

print(textguten)

from nltk.book import *

text1.concordance("monstrous")
text1.similar("monstrous")

text2.similar("monstrous")

text2.common_contexts(["monstrous","very"])

text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])

text1.count("monstrous")

print("length of text:"+str(len(text3)))

sorted(set(text3))

print("length of token:"+str(len(set(text3))))

#to find how many % of unique words in the text
text='nikhil aivalli aivalli nikhil'
text=word_tokenize(text)

def lexical_richness(text):
   return (len(set(text))/len(text))*100
    
lexical_richness(text)

# %of occurence of a word

def fn(text,word):
   return (text.count(word)/len(text))*100
    
fn(text,"aivalli")


#freequency distribution
#used to classify documents(Eg: sports,national news)
fdist=FreqDist(text1)
fdist

len(fdist)
fdist.most_common(4)
fdist.plot(12,cumulative=True)
#################

nltk.corpus.gutenberg.fileids()
emma=nltk.corpus.gutenberg.words('austen-emma.txt')
emma=nltk.Text(emma)
emma.concordance("surprize")
#############################

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

nltk.corpus.state_union.fileids()
trained=state_union.raw('2005-GWBush.txt')
tested=state_union.raw('2006-GWBush.txt')
#training PunktSentenceTokenizer below

custom_sent_tokenizer=PunktSentenceTokenizer(trained)
tokenized=custom_sent_tokenize.tokenize(tested)








#######################

import requests
f_online='http://www.gutenberg.org/files/11111/11111.txt'
f_rawtext=requests.get(f_online).text
print(f_rawtext)

raw_words = word_tokenize(f_rawtext[:100])

tags=nltk.pos_tag(raw_words,tagset='universal')
print(tags[:9])

###################################################################3

#tagging

text="Hi how are you"
w=word_tokenize(text)
for i in w:
    tagged=nltk.pos_tag(i)
    print(tagged)

##########################################################33
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

nltk.corpus.state_union.fileids()
trained=state_union.raw('2005-GWBush.txt')
tested=state_union.raw('2006-GWBush.txt')
#training PunktSentenceTokenizer below

custom_sent_tokenizer=PunktSentenceTokenizer(trained)
tokenized=custom_sent_tokenizer.tokenize(tested)

def text_processing():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk:{<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser=nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            print(chunked)
            
            for subtree in chunked.subtrees():
                print(subtree)
            chunked.draw()   
    except Exception as e:
        print(str(e))
text_processing()


        
