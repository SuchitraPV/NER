import nltk #importing NLTK package to perform NLP operations
import csv

csvFile = open('news.csv', encoding='utf-8').readlines() #Reading the news.csv file

def process_Paragraphs(csvFile):  #a function to perform NER
    count=0
    for line in csvFile:         #taking a line at a time we tokenize, perform Parts-of-Speech Tagging and then do chunking to find NER using the popular nltk.chuck.api module operations
        if count==0:            #skipping the first line which contains field names
            count+=1
        else:
            tokenized = nltk.word_tokenize(line)
            POS_tagged = nltk.pos_tag(tokenized)    # word tokenization and POS tagging to the para
            NER_entity = nltk.ne_chunk(POS_tagged)    #POS tags are added
            NER_entity.draw()                     #the tree representation of the NER
            count+=1

process_Paragraphs(csvFile)           #Performing NER for news.txt as input
