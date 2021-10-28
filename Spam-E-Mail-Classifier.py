#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# - The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
# 
# - The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
# 

# ![image.png](attachment:image.png)

# # STEP #0: LIBRARIES IMPORT
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # STEP #1: IMPORT DATASET

# In[2]:


spam_df = pd.read_csv("emails.csv")


# In[3]:


spam_df.head(10)


# In[4]:


spam_df.tail()


# In[5]:


spam_df.describe()


# In[6]:


spam_df.info()


# # STEP #2: VISUALIZE DATASET

# In[7]:


# Let's see which message is the most popular ham/spam message
spam_df.groupby('spam').describe()


# In[38]:


# Let's get the length of the messages
spam_df['length'] = spam_df['text'].apply(len)
spam_df.head()


# In[39]:


spam_df


# In[42]:


spam_df['length'].plot(bins=100, kind='hist') 


# In[43]:


spam_df.length.describe()


# In[47]:


# Let's see the longest message 43952
spam_df[spam_df['length'] == 43952]['text'].iloc[0]


# In[48]:


# Let's divide the messages into spam and ham


# In[49]:


ham = spam_df[spam_df['spam']==0]


# In[50]:


spam = spam_df[spam_df['spam']==1]


# In[51]:


ham


# In[52]:


spam


# In[58]:


spam['length'].plot(bins=60, kind='hist') 


# In[57]:


ham['length'].plot(bins=60, kind='hist') 


# In[59]:


print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")


# In[60]:


print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")


# In[61]:


sns.countplot(spam_df['spam'], label = "Count") 


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# # STEP 3.1 EXERCISE: REMOVE PUNCTUATION

# In[62]:


import string
string.punctuation


# In[63]:


Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'


# In[65]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[68]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# # STEP 3.2 EXERCISE: REMOVE STOPWORDS

# In[70]:


# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


# In[72]:


Test_punc_removed_join


# In[71]:


Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[73]:


Test_punc_removed_join_clean # Only important (no so common) words are left


# # STEP 3.3 EXERCISE: COUNT VECTORIZER EXAMPLE 

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[16]:


print(vectorizer.get_feature_names())


# In[17]:


print(X.toarray())  


# # LET'S APPLY THE PREVIOUS THREE PROCESSES TO OUR SPAM/HAM EXAMPLE

# In[74]:


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[85]:


# Let's test the newly added function
spam_df_clean = spam_df['text'].apply(message_cleaning)


# In[87]:


print(spam_df_clean[0])


# In[82]:


print(spam_df['text'][0])


# # LET'S APPLY COUNT VECTORIZER TO OUR MESSAGES LIST

# In[88]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])


# In[89]:


print(vectorizer.get_feature_names())


# In[90]:


print(spamham_countvectorizer.toarray())  


# In[91]:


spamham_countvectorizer.shape


# # STEP#4: TRAINING THE MODEL WITH ALL DATASET

# In[92]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = spam_df['spam'].values
NB_classifier.fit(spamham_countvectorizer, label)


# In[93]:


testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


# In[94]:


test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# In[98]:


# Mini Challenge!
testing_sample = ['Hello, I am Ryan, I would like to book a hotel in Bali by January 24th', 'money viagara!!!!!']


# In[99]:


testing_sample = ['money viagara!!!!!', "Hello, I am Ryan, I would like to book a hotel in SF by January 24th"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# # STEP#4: DIVIDE THE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING

# In[100]:


X = spamham_countvectorizer
y = label


# In[101]:


X.shape


# In[102]:


y.shape


# In[103]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[104]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[105]:


# from sklearn.naive_bayes import GaussianNB 
# NB_classifier = GaussianNB()
# NB_classifier.fit(X_train, y_train)


# # STEP#5: EVALUATING THE MODEL 

# In[106]:


from sklearn.metrics import classification_report, confusion_matrix


# In[107]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[108]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[109]:


print(classification_report(y_test, y_predict_test))


# # STEP #6: LET'S ADD ADDITIONAL FEATURE TF-IDF

# - Tf–idf stands for "Term Frequency–Inverse Document Frequency" is a numerical statistic used to reflect how important a word is to a document in a collection or corpus of documents. 
# - TFIDF is used as a weighting factor during text search processes and text mining.
# - The intuition behing the TFIDF is as follows: if a word appears several times in a given document, this word might be meaningful (more important) than other words that appeared fewer times in the same document. However, if a given word appeared several times in a given document but also appeared many times in other documents, there is a probability that this word might be common frequent word such as 'I' 'am'..etc. (not really important or meaningful!).
# 
# 
# - TF: Term Frequency is used to measure the frequency of term occurrence in a document: 
#     - TF(word) = Number of times the 'word' appears in a document / Total number of terms in the document
# - IDF: Inverse Document Frequency is used to measure how important a term is: 
#     - IDF(word) = log_e(Total number of documents / Number of documents with the term 'word' in it).
# 
# - Example: Let's assume we have a document that contains 1000 words and the term “John” appeared 20 times, the Term-Frequency for the word 'John' can be calculated as follows:
#     - TF|john = 20/1000 = 0.02
# 
# - Let's calculate the IDF (inverse document frequency) of the word 'john' assuming that it appears 50,000 times in a 1,000,000 million documents (corpus). 
#     - IDF|john = log (1,000,000/50,000) = 1.3
# 
# - Therefore the overall weight of the word 'john' is as follows 
#     - TF-IDF|john = 0.02 * 1.3 = 0.026

# In[142]:


spamham_countvectorizer


# In[143]:


from sklearn.feature_extraction.text import TfidfTransformer

emails_tfidf = TfidfTransformer().fit_transform(spamham_countvectorizer)
print(emails_tfidf.shape)


# In[144]:


print(emails_tfidf[:,:])
# Sparse matrix with all the values of IF-IDF


# In[147]:


X = emails_tfidf
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[148]:


print(classification_report(y_test, y_predict_test))

