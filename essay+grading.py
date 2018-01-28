
# coding: utf-8

# In[2]:

import tensorflow as tf


# In[1]:

import nltk
import random
from nltk.book import *


# ## Features definition

# In[3]:

def avg_word_length(essay):
    s=0
    numWords=0
    for word in essay.split(' '):
        s+=len(word)
        numWords+=1
    return s/numWords


# In[4]:

def numI(essay):
    count=0
    for word in essay.split(' '):
        if word=='I':
            count+=1
    return count


# In[5]:

def numWords(essay):
    count=0
    for word in essay.split(' '):
        count+=1
    return count


# In[6]:

def uniqueWords(essay):
    d={}
    for word in essay.split(' '):
        if word not in d:
            d[word]=0
    return len(d)


# In[7]:

def sentenceLength(essay):
    s=0
    numSent=0
    for sent in essay.split('.'):
        numWords=0
        for word in sent.split(' '):
            numWords+=1
        s+=numWords
        numSent+=1
    return s/numSent


# In[8]:

def captErrors(essay):
    errors=0
    sents=essay.split('.')
    for i in range(len(sents)-1):
        sent=sents[i]
        if len(sent)>2:
            if sent[1]==sent[1].lower():
                errors+=1
    return errors


# In[9]:

def essay_features(essay):
    features={}
    #features['length']=numWords(essay) # decreases accuracy!!!
    features['wordLength']=avg_word_length(essay)
    features['numI']=numI(essay)
    features['uniqueWords']=uniqueWords(essay)
    features['sentenceLength']=sentenceLength(essay)
    features['captErrors']=captErrors(essay)
    return features


# # Processing training/test sets

# In[10]:

import pandas as pd
df = pd.read_excel('training_set_rel3.xls', usecols=[2,6])


# In[11]:

good=[]
bad=[]
for i in df.index:
    if df['domain1_score'][i]>9:
        good.append(df['essay'][i])
    else:
        bad.append(df['essay'][i])
labeled_essays = [(essay, 'good') for essay in good]+[(essay, 'bad') for essay in bad]
random.shuffle(labeled_essays)
featuresets = [(essay_features(n), clas) for (n, clas) in labeled_essays]
essays=[(n, clas) for (n, clas) in labeled_essays]
essay_train_set, essay_test_set = featuresets[400:], featuresets[:400]


# # Decision tree

# In[12]:

classifier = nltk.classify.DecisionTreeClassifier.train(essay_train_set, entropy_cutoff=0, support_cutoff=0)


# In[16]:

print(essay_test_set[0])
print(essays[0])
print(labeled_essays[0])


# In[54]:

c=0
t=0
for i in range(len(essay_test_set)):
    essayFeatures=essay_test_set[i]
    essay=essays[i]
    clas=classifier.classify(essayFeatures[0])
    t+=1
    if clas!=essayFeatures[1]:
        print(i)
        print('prediction:',clas,'actual:',essayFeatures[1])
        print(essayFeatures[0])
        print(essay)
        print('\n')
        c+=1
print('number misclassified:',c)
print('total:',t)
        


# Accuracy

# In[13]:

nltk.classify.accuracy(classifier, essay_test_set)


# # SVM
import nltk.classify
from sklearn.svm import LinearSVC

classifier2 = nltk.classify.SklearnClassifier(LinearSVC())
classifier2.train(essay_train_set)
# Accuracy
nltk.classify.accuracy(classifier2, essay_test_set)