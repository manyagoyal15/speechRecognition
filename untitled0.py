# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:20:23 2020

@author: welcome
"""

import pandas as pd
df = pd.readcsv('data/movie_data.csv')
df.head(10)
df['review'][0]

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

docs = np.array(['The sun is shining',
                'The weather is sweet',
                'The sun is shining, the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)

print(tfidf.fit_transform(bag).toarray())

df.loc[0, 'review'][-50:]

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

preprocessor(df.loc[0, 'review'][-50:])

preprocessor("</a> This :) is a :( test :-)!")

df['review'] = df['review'].apply(preprocessor)

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer('runners like running and thus they run')

tokenizer_porter('runners like running and thus they run')

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner like running and runs a lot')[-10:] if w not in stop]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None,
                       lowercase=False,
                       preprocessor=None,
                       tokenizer=tokenizer_porter,
                       used_idf=True,
                       norm='l2',
                       smooth_idf=True)

y=df.sentiment.values
X=tfidf.fit_transform(df.review)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5, shuffle=False) 

import pickle
from sklearn.limear_model import LogisticRegressionCV

clf = LogisticsRegressionCV(cv=5,
                           scoring='accuracy',
                           random_state=0,
                           n_jobs=-1,
                           verbose=3,
                           max_iter=300).fit(X_train, y_train)

saved_model = open('saved_model.sav', 'wb')
pickle.dump(clf, saved_model)
saved_model.close()
