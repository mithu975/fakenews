import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

#reading data into pandas dataframe
df=pd.read_csv('news.csv')

#printing first 5 entries in the table/dataframe
print(df.head())

#deciding predicate and predicator variables
x=df['text']
y=df['label']

#splitting data into train dataset and test dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#vectorization and fitting data 
tf_vect=TfidfVectorizer(stop_words='english',max_df=0.6)
tf_train=tf_vect.fit_transform(x_train) 
tf_test=tf_vect.transform(x_test)

#predicting lables of test data using PassiveAgrressiveClassifier
passiveac=PassiveAggressiveClassifier(max_iter=60) #more the value of max_iter, more accuracy can be obtained
passiveac.fit(tf_train,y_train)
pred=passiveac.predict(tf_test)

#calculating accuracy

print(classification_report(y_test, pred))

print(confusion_matrix(y_test,pred))

print(accuracy_score(y_test,pred))

#Fake News classifier model 