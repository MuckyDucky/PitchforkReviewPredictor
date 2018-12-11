from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import itertools
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
#%matplotlib inline
import os
init_notebook_mode()


input_file = os.getcwd() + "\p4kreviews.csv"

#print (input_file)

reviews = pd.read_csv(input_file, header = 0, engine="python")
stWords = stopwords.words('english')


def cleanme(txt):
    sent = txt.lower()
    wrds = word_tokenize(sent)
    #wrds = 'blah'
    clwrds = [w for w in wrds if not w in stWords]
    ln = len(clwrds)
    pos = pd.DataFrame(pos_tag(wrds))
    pos = " ".join(list(pos[pos[1].str.contains("JJ")].iloc[:,0]))
    rt = [ln, " ".join(clwrds), pos]
    return rt



tmp = list()
for i in range(5):
	if type(reviews.iloc[i]['review']) == str:
		print(round((i/len(reviews))*100,2),'%')
		tmp.append(cleanme(reviews.iloc[i]['review']))
	# if type(tmp.append(len(reviews.iloc[i]['review']))) != str:
	# 	print(i)
	#tmp.append(len(reviews.iloc[i]['review']))
tmp = pd.DataFrame(tmp)
tmp.columns = ['reviewlen', 'cleanrev', 'adjreview']

#Add calculated columns back to the dataset
reviews = reviews.reset_index()
reviews = pd.concat([reviews,tmp], axis=1)
print(reviews.head())

plt2 = go.Histogram(x = reviews.reviewlen)
lyt2 = go.Layout(title="Frequency of Review Length", xaxis=dict(title='Review Length', range=[0,400]), yaxis=dict(title='Frequency'))
fig2 = go.Figure(data=[plt2], layout=lyt2)

#plotly.offline.plot(fig2, filename='file.html')

reviews = reviews.sort_values(by='reviewlen')
plt3 = go.Scatter(x = reviews.reviewlen, y = reviews.score, mode='markers')
lyt3 = go.Layout(title="Review Length vs. Score", xaxis=dict(title='Review Length'),yaxis=dict(title='Rating'))
fig3 = go.Figure(data=[plt3], layout=lyt3)
plotly.offline.plot(fig3, filename='file.html')
#iplot(fig3)
print("Review Length to Rating Correlation:",reviews.reviewlen.corr(reviews.reviewsrating))

#iplot(fig2)