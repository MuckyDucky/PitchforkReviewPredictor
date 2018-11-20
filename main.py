import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from math import floor
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


input_file = os.getcwd() + "\p4kreviews.csv"

print (input_file)

reviews = pd.read_csv(input_file, header = 0, engine="python")
original_headers=list(reviews.columns.values)
reviews_noheader = reviews[1:]

#number of rows


#select train data

#select only the album title, artist, and score column (for regression practice)
scores_only=pd.DataFrame()
scores_only=reviews_noheader[['album','artist','score']]
scores_only.set_index("album",inplace=True)
#print(edited.head())
#print(scores_only.loc["In Rainbows"])

n_rows = scores_only.shape[0]
###Average score of artist albums
artist_mean_score = reviews_noheader.groupby("artist",as_index=False).score.mean()
artist_mean_score.set_index("artist",inplace=True)
#print(artist_mean_score.head())
#print

#scores_only["a"]=artist_mean_score.loc[scores_only["artist"]]
#print(artist_mean_score.iloc[0].loc["score"])


scores_only['artist_average']=scores_only['artist'].map(artist_mean_score['score'])
print(scores_only.head())

#print(artist_mean_score.loc['Minnie Riperton'])


# for row_n in range(0,3):
# 	print(artist_mean_score.loc[scores_only.iloc[row_n]["artist"]])




#print(scores_only.head())

n_rows = scores_only.shape[0]
n_columns = scores_only.shape[1]
seventy_percent = floor(n_rows * .7)

# #x = artist_average
# scores_only_X_train = scores_only.iloc[:-seventy_percent]['artist_average']
# scores_only_X_test = scores_only.iloc[-seventy_percent:]['artist_average']
#
# print(scores_only_X_test.head())
#
# #y = score
# scores_only_Y_train = scores_only.iloc[:-seventy_percent]['score']
# scores_only_Y_test = scores_only.iloc[-seventy_percent:]['score']
#
# #print(scores_only_test.head())

#x = artist_average
scores_only_X_train = scores_only.iloc[201:300]['artist_average'].to_frame()
scores_only_X_test = scores_only.iloc[1:100]['artist_average'].to_frame()

#print(scores_only_X_test.head())

#y = score
scores_only_Y_train = scores_only.iloc[201:300]['score'].to_frame()
scores_only_Y_test = scores_only.iloc[1:100]['score'].to_frame()


#Create linear regression object
regr = linear_model.LinearRegression()

#Train the model using traning set
regr.fit(scores_only_X_train,scores_only_Y_train)
#regr.fit(df2.iloc[1:1000, 5].to_frame(), df2.iloc[1:1000, 2].to_frame())

diabetes_y_pred = regr.predict(scores_only_X_train)
#
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(scores_only_Y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(scores_only_Y_test, diabetes_y_pred))


# Plot outputs
plt.scatter(scores_only_X_test, scores_only_Y_test,  color='black')
plt.plot(scores_only_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


#FEATURES


##Numerical



###Recent score of artist albums


###Score of artists’ first album on site

#Genre





#print(s[0])


#diabetes_X = reviews_noheader[:,np.newaxis,2]

#print(reviews_noheader[1:4])




