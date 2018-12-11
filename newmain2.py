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
import re
init_notebook_mode()


input_file = os.getcwd() + "\p4kreviews.csv"

#print (input_file)

reviews = pd.read_csv(input_file, header = 0, engine="python")
stWords = stopwords.words('english')


def cleanme(txt):
    sent = txt.lower()
    sent = re.sub('[^A-Za-z ]', '', sent)
    wrds = word_tokenize(sent)
    clwrds = [w for w in wrds if not w in stWords]
    ln = len(clwrds)
    rt = [ln, " ".join(clwrds)]
    return rt

limit = 10000
print('processing reviews number of: ', limit)
newlimit = limit

tmp = list()
for i in range(limit):
    if type(reviews.iloc[i]['review']) == str:
        percentage = round((i/limit)*100,2)
        if percentage%1 == 0:
            print(percentage,'%')
        #tmp.append(cleanme(reviews.iloc[i]['review']))
        listtoappend=cleanme(reviews.iloc[i]['review'])
        rounded_review_sc = round(reviews.iloc[i]['score'])
        listtoappend.append(rounded_review_sc)
        tmp.append(listtoappend)
    else:
        print('error in i:' , i)
        print('album:', reviews.iloc[i]['artist'])
        print('artist:', reviews.iloc[i]['artist'])
        print('review:' , reviews.iloc[i]['review'])


	# if type(tmp.append(len(reviews.iloc[i]['review']))) != str:
	# 	print(i)
	#tmp.append(len(reviews.iloc[i]['review']))
tmp = pd.DataFrame(tmp)
tmp.columns = ['reviewlen', 'cleanrev','intscore']

#Add calculated columns back to the dataset
reviews = reviews.reset_index()
reviews = pd.concat([reviews,tmp], axis=1)
print(reviews.head())

plt2 = go.Histogram(x = reviews.reviewlen)
lyt2 = go.Layout(title="Frequency of Review Length", xaxis=dict(title='Review Length', range=[0,400]), yaxis=dict(title='Frequency'))
fig2 = go.Figure(data=[plt2], layout=lyt2)

#plotly.offline.plot(fig2, filename='file.html')

reviews = reviews.sort_values(by='reviewlen')
plt3 = go.Scatter(x = reviews.reviewlen, y = reviews.intscore, mode='markers')
lyt3 = go.Layout(title="Review Length vs. Score", xaxis=dict(title='Review Length'),yaxis=dict(title='Rating'))
fig3 = go.Figure(data=[plt3], layout=lyt3)
#plotly.offline.plot(fig3, filename='file.html')
#iplot(fig3)
print("Review Length to Rating Correlation:",reviews.reviewlen.corr(reviews.score))

#iplot(fig2)


#Setting up the X and Y data, where X is the review text and Y is the rating
#Three different inputs will be used: original review text, cleaned review text, and only adjectives review text
x1 = reviews.review[:limit]
x2 = reviews.cleanrev[:limit]
#x3 = reviews.adjreview[:newlimit]
y = reviews.intscore[:limit]


#Creating a vectorizer to split the text into unigrams and bigrams
vect = TfidfVectorizer(ngram_range = (1,2),min_df=0.2)
x_vect1 = vect.fit_transform(x1.astype('U'))
x_vect2 = vect.fit_transform(x2.astype('U'))
#x_vect3 = vect.fit_transform(x3.astype('U'))


#Making some simple functions for linear svc, knn, and naive bayes
def linsvc(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 10)
    classf = LinearSVC()
    classf.fit(x_train, y_train)
    pred = classf.predict(x_test)
    print("Linear SVC:",accuracy_score(y_test, pred))
    return(y_test, pred)

def revknn(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 10)
    classf = KNeighborsClassifier(n_neighbors=2)
    classf.fit(x_train, y_train)
    pred = classf.predict(x_test)
    print("kNN:",accuracy_score(y_test, pred))
    return(y_test, pred)

def revnb(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 10)
    classf = MultinomialNB()
    classf.fit(x_train, y_train)
    pred = classf.predict(x_test)
    print("Naive Bayes:",accuracy_score(y_test, pred))
    return(y_test, pred)


svmy1,svmp1 = linsvc(x_vect1,y)
svmy2,svmp2 = linsvc(x_vect2,y)
#svmy3,svmp3 = linsvc(x_vect3,y)

knny1,knnp1 = revknn(x_vect1,y)
knny2,knnp2 = revknn(x_vect2,y)
#knny3,knnp3 = revknn(x_vect3,y)

nby1,nbp1 = revnb(x_vect1,y)
nby2,nbp2 = revnb(x_vect2,y)
#nby3,nbp3 = revnb(x_vect3,y)


#This function will plot a confusion matrix and is taken from the sklearn documentation with just some minor tweaks
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]),decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

c1 = confusion_matrix(svmy1,svmp1)
c2 = confusion_matrix(svmy2,svmp2)
c3 = confusion_matrix(nby2,nbp2)
class_names = ['1', '2', '3', '4', '5','6','7','8','9','10']
plt.figure()
plot_confusion_matrix(c1, classes=class_names,normalize=False,title='Confusion matrix - SVM Full Review')
plt.figure()
plot_confusion_matrix(c2, classes=class_names,normalize=False,title='Confusion matrix - SVM No Stopwords')
plt.figure()
plot_confusion_matrix(c3, classes=class_names,normalize=False,title='Confusion matrix - NB No Stopwords')