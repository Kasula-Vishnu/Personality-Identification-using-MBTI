import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
import nltk
import pickle
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from imblearn.over_sampling import  SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("mbti_1.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
print(dataset.head)
dataset.tail

stop_words = stopwords.words('english')
ps = PorterStemmer()
corpus = []
for i in range(0, len(dataset)):
  review = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ',dataset['posts'][i])
  review = re.sub('[^a-zA-Z]', ' ', review)
  review = review.lower() 
  words = nltk.word_tokenize(review)
  words = [ps.stem(word) for word in words if word not in set(stop_words)]
  review = ' '.join(words)  
  corpus.append(review)

dataset.posts = corpus

dataset.posts[0]


enc = LabelEncoder()
dataset['new type'] = enc.fit_transform(dataset['type'])

X = dataset['new type']

X_train, X_test, Y_train, Y_test = train_test_split(corpus, X, test_size=0.2,  random_state=42)
print ((X_train.shape),(Y_train.shape),(X_test.shape),(Y_test.shape))

cv = TfidfVectorizer( ngram_range=(1, 1), max_features=5000)
X_train = cv.fit_transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()

print(X_train.shape)

'''accuracies = {}
XGB = XGBClassifier()
XGB.fit(X_train,Y_train)

Y_pred = XGB.predict(X_test)
predictions = [round(value) for value in Y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
accuracies['XG Boost'] = accuracy* 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))'''


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
accuracies = {}
logreg = LogisticRegression(C=0.5, class_weight='balanced',
                            fit_intercept=True,intercept_scaling=1,
                            multi_class='ovr', n_jobs=-1)
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
predictions = [round(value) for value in Y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
accuracies['Logistic Regression'] = accuracy* 100.0
print("Accuracy: %.2f%%" % (accuracy * 100.0))

def input_preprocesing(text):
  filter= [ ]
  review = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ',text)
  review = re.sub('[^a-zA-Z]', ' ', review)
  review = review.lower()
  stop_words = set(stopwords.words("english"))
  word_tokens = word_tokenize(review)
  filtered_text = [word for word in word_tokens if word not in stop_words]
  
  return filtered_text


text = "My name is Vishnu and Iam from Guntakal"
lol = input_preprocesing(text)
print(lol)

def tfidf_vectorizer(preprocessed_text):
  
  Y = cv.transform(preprocessed_text).toarray()
  return Y

vectorized_text = tfidf_vectorizer(lol)

print(vectorized_text.shape)

prediction = logreg.predict(vectorized_text)[0]
print(prediction)


pickle.dump(cv, open('vectorizer.pkl', 'wb'))
'''pickle.dump(XGB,open('model_XGB.pkl', 'wb'))'''
pickle.dump(logreg,open('model_logreg.pkl', 'wb'))

