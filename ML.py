# intermediate + ML
""" you want to maintain readability + when you scale the project, 
	these concepts make them easier. """

# pip install sklearn
# pip install quandl
# pip install pandas

# regression
import pandas as pd 
import Quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# stock price (any) is a continuous data
# get the Quandl data
df = Quandl.get('WIKI/GOOGL')
print(df.head())

# select the columns
df = df[["Adj. Open", "Adj. High", "Adj. Close", "Adj. Volume"]]
# define a new column
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0
df["PCT_change"] = df["HL_PCT"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

# reselect columns needed
# we have features
df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]
print(df.head())

# regression features and labels

# work with DataFrame "df"
# set a variable
forecast_col = "Adj. Close"
# in ML you can't work with NaN data but you don't want to throw away data
df.fillna(-99999, inplace = True)

# math.ceil rounds off to the nearest value and we make it an integer value 
# because math.ceil returns float value
# we predict 10% of the Data Frame, feel free to change the 0.1
forecast_out = int(math.ceil(0.01 * len(df)))

# we are shifting the columns negatively
# shift in pandas shift index by desired number of periods with an optional time freq
df["label"] = df[forecast_col].shift(-forecast_out)
""" The label column in df is the forecast column (aka Adj. Close) shifted by 16 days """

df.dropna(inplace = True)
print(df.head())

# feature X and label lowercase y
# features are everything except for the label column
# df.drop returns new data frame
# note: axis = 1 denotes that we are referring to a column, not a row
X = np.array(df.drop(["label"]), 1)
y = np.array(df["label"])

# we scale X before we feed it through classifier
# you have to scale it/normalized with all the other data: you have to include training data
# scale along side all the other values
X = preprocessing.scale(X)

# this code is to have the X values where we have for y values because we shifted earlier
#X = X[:-forecast_out + 1]
y = np.array(df["label"])
print(len(x), len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

clf = LinearRegression()
# if we want to use svm? 
# just change the clf = svm.SVR()

# train
clf.fit(X_train, y_train)
# test: we are testing first
accuracy = clf.score(X_test, y_test)
print(accuracy) # 0.96 == 96% accuracy?

# MSE : of estimator measures the average of the squares of the errors of deviation
# the difference between the estimator and what is estimated.
# with LR accuracy is the MSE

clf1 = svm.SVR(kernel = "poly")
# what's kernel be explained in SVM

""" the above example shows how easy it is to switch the algorithms """

X = np.array(df.drop["label"], 1)
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# re write code
import pandas as pd
import numpy as np
import Quandl
import math, datetime
from sklearn import preprocessing,cross_validation,svm,linear_model
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

quandl.ApiConfig.api_key = 'xbPnXMAxKyEVqwzW9TWv'
df  = quandl.get_table('WIKI/PRICES')

print(df.head())

df['HL_PCT'] =(df['adj_high']-df['adj_close'])/df['adj_close'] * 100.0
df['PCT_change'] =(df['adj_close']-df['adj_open']) / df['adj_open'] * 100.0
df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]
forecast_col = df['adj_close']
forecast_out = int(math.ceil(0.001*len(forecast_col)))
df['label'] = forecast_col.shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))#,'adj_close'],1))

X = preprocessing.scale(X)
#print("X after preprocessing.scale ",X)
X_lately = X[-forecast_out:]
#print("X_lately",X_lately)
X = X[:-forecast_out]

#print(df)
#print("X",X)
Y = np.array(df['label'])
#Y=preprocessing.scale(Y)
Y = Y[:-forecast_out]
#print("Y ",Y)

x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,Y, test_size=0.2)
clf = linear_model.LinearRegression(n_jobs=-1)
clf = clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test) 

forecast_set = clf.predict(X_lately)
#forecast_set_whole=clf.predict(X)
print(accuracy, forecast_set)#,forecast_set_whole)
forecast_set=np.array(forecast_set)

df['Forecast'] = np.nan # entire column is NaN specifying

last_date = df.iloc[-1].name # getting the name of it
last_unix = last_date #.timestamp()
one_day = 1
next_unix = last_unix + one_day

for i in forecast_set:
 next_date = next_unix #datetime.datetime.fromtimestamp(next_unix)
 next_unix += one_day
 df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.label[:-forecast_out])
print(forecast_set)
df['Adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4) # location 4
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# ---------------------------------------------------------------------------
# 6. pickling and scaling 

# pickle == serialization
import pickle

with open("linearregression.pickle", "wb") as f:

	# dumpbs the classifer in the "f"
	pickle.dump(clf, f)

pickle_in = open("linearregression.pickle", "rb")
clf = pickle.load(pickle_in)


# -------------------------------------------------------------------------------
# part7. regression how it works?

# part8. how to program Best fit slope?

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# setting numpy array and data type explicitly
xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

plt.scatter(xs, ys)
plt.show()

def best_fit_slope(xs, ys):

	m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
		
		# order of parentheses must get right!!!!
		#(mean(xs)^2)) this gives data type error Int, Float64
		#(mean(xs)**2))
		   ((mean(xs) * mean(xs)) - mean(xs*xs)) )
	return m

m = best_fit_slope(xs, ys)
print(m)

# -------------------------------------------------------------------------------

""" quick thoughts on boolean methods in Python, R, Julia:

	Python : True, False
	R: TRUE, FALSE
	Julia: true, false
"""

# -------------------------------------------------------------------------------

# part 9. best fit line programming
# we need to calculate y-intercept of best fit line
# b = y - mx (y and x has - above)
# we just need to add 'b' in the same function
# just follow exactly the equations

import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

def best_fit_slope_and_intercept(xs, ys):

	m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
		
		# order of parentheses must get right!!!!
		#(mean(xs)^2)) this gives data type error Int, Float64
		#(mean(xs)**2))
		   ((mean(xs) * mean(xs)) - mean(xs*xs)) )

	b = mean(ys) - m*mean(xs)

	# returns two variables
	return m, b

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

# list comprehension: 
# we have values for 'm' and 'b' from the above function
# a list of y's

# e.g. when x = 1; y-intercept = 3.5
#      when x = 2; y-intercept = 4.5
#      when x = 3; y-intercept = 5.6 etc
regression_line = [(m*x) + b for x in xs]

# same as
# regression_line = []
# for x in xs:
# 	regression_line.append((m*x) + b)

plt.scatter(xs, ys)
plt.plot(xs, regression_line) # regression lines are the list of Y's
plt.show()

# what if we want to make predictions?
predict_x = 8

# when x is 8 what is y?
predict_y = (m*predict_x) + b
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.scatter(precict_x, predict_y, color = "g")
plt.show()

# best fit line with good fit line?
# how good is this best fit line? How accurate is the best fit line? 
# we can calculate how good the best fit line is?

# -------------------------------------------------------------------------------

# Part 10.
# R squared theory

""" 
	 We calculated best fit line, but the question is how accurate are they?
	 How to calculate accuracy of the best fit line? We can do that by 
	 "R squared/Coefficient of Determination". CoD is calculated using 
	 Squared Error.

	 Error is the distance between the estimated and the best fit line
	 We square it so we are only dealing with Positive values: thus 
	 R2 theory

	How to then determine R2: Coefficient of Determination? 
	R^2 = 1 - SEy^ (y^ is same as best fit, regression line same thing)
	     -----------
	     SEy- (-mean)

	
	After we calculated R2 what are good and bad values?
	
	If R^2 = 0.8 
	SEy^ = 2
	SEy- = 10

	We saying here that SEy^ line is significantly lower than the
	SEy- (mean of the line)

	This is good thing, we preferably want it to be lower than that
	but this means that the data is probably linear
	thus R^2 = 0.8 pretty good! 

	If R^2 is 0.3?
	SEy^ to be 0.7
	SEy- to be 10
	now a bit closer thus, we want the R^2 to be higher so that the model 
	is more linear.

	[ This R^2 is coefficient of determination!!! ]

"""
# -------------------------------------------------------------------------------

# Part 11.
# Programming R squared

# function that calculated squared error (SE)
 
""" Recall that the formula for working out R^2 / Coefficient of Determination is

	R^2 = 1 - (SEy^/SEy-) """

# this works out SEy^
# SEy^ is difference between line of best fit and the actual point
def squared_error(ys_orig, ys_line):

	""" R^2 : squared error.
		Re-call that the SE is distance between distance between
		the line and the actual point + Squared. """

	# this is squared error for entire line
	return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):

	""" R^2 = 1 - (SEy^/SEy-) """

	# test this in Python
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	""" this line of code above basically does this: 

		for how many y values in y_orig, those values must be 
		mean(y_orig) such as 

		[mean(y_orig) for y in y_orig]
		# outputs
		[7.5, 7.5, 7.5, 7.5, 7.5] etc. """

	squared_error_reg = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_reg / squared_error_y_mean)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)
# outputs 0.58 

# some theoractical background information

""" The correlation coefficient formula will tell you how strong of a linear
	relationship there is between two variables. R^2 is the square of the 
	correlation coefficient.

	Coefficient of determination gives you an idea of how many data points fall
	within the results of the line formed by the regression equation. The higher
	the coefficient, the higher percentage of points the line passes through 
	when the data points and line are plotted. If the coefficient is 0.80, 80% of
	the points should fall within the regression line. Values of 1/0 indicates 
	the regression line represents all or none of the data. A higher coefficient
	is an indicator of a better goodness of fit for the observations. 

"""
# -------------------------------------------------------------------------------

# part 12. Testing Assumptions

""" two algorithms R2, equation of best fit line: 
	
	Unit testing is required. This is not gonna be unit test but idea is same.
	We could work with sample data and test if it works. To make sure our thing
	works finely. """

# we will use random numbers
# this is continuation from the last snippet of codes
import random
import numpy as np

def create_dataset(how_many, variance, step = 2, correlation = False):

	val = 1
	ys = []

	for i in range(hm):
		# from - variance to + variance
		y = val + random.randrange(-variance, variance)
		ys.append(y)

		if correlation and correlation == "pos":
			val += step
		elif correlation and correlation == "neg":
			val -= step

	# list comprehension to assign values to a list 'xs'
	xs = [i for i in range(len(ys))]

	# create_datset function returns 2 np.array
	return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

# create a new data set
xs, ys = create_dataset(40, 40, 2, correlation = "pos")

# run in conjunction with the other snippet codes

import matplotlib.pyplot as plt

# this plots the point as a scatter
plt.scatter(xs, ys)

# this plots the regression line
plt.plot(xs, regression_line)

# prediction data point
plt.scatter(predict_x, predict_y, s = 100, color = "g") # s is a size
plt.show()

# if we decrease the variance the plots will be tighter
# coefficient of determination is also high

# only adjust the values in the variance and you will see the difference
xs, ys = create_dataset(40, 10, 2, correlation = "pos")

# -------------------------------------------------------------------------------
# END OF REGRESSION

# p13.
# Classification w/ K Nearest Neighbors
# K Nearest Neighbors
# Some theoretical backgrounds:

""" objective of linear regression was to create a model that best fits the data
	objective of classification is to create a model that best divides the data
"""

# KNN - who is the closest from the data points?
# what is k?
# you decide what the number of k is

""" 
	if k = 2, for unknown data point in the plotting figure,
	you find two closest neighbors to k. With k = 2, you got 2 points
	close to each other. 2 Closest points by k = 2. You'd want to have the k = odd number
	It's a super simple algorithm. what if you have 3 groups? k = 3? is not good idea?
	If you got 3 groups you need to k = 5 at least to avoid split vote.

	What's needed for k means is you can get actual classification
	you can get both accuracy in the model so you can test and train for overall
	accuracy but each point also can have degree of confidence

	- - + : 66% -> confidence

	Some downfalls of kNN?:
	The measurement of distance is the simple Euclidean Distance: distance between
	any given point and all of the other points, and what are the closest three? or
	k = ?
	
	Thus,
	Larger the data set the worse the algorithm runs. SVM is much more efficient
	when it comes to classification problem. k-means not really training.

	# Scaling is not so good in k-means, but SVM scales much better!
"""
# -------------------------------------------------------------------------------

# p14.
# kNN application
# kNN - new data point is applied and use euclidean distance to compute the value
# UCI dataset is used:
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data﻿

# add header to the datase downloaded
# id, clump_thickness, unif_cell_size, unif_cell_shape, marg_adhesion, single_epith_cell_size, bare_nuclei, bland_chrom, norm_nuceloli, mitosis, class
#  1000025,5,1,1,1,2,1,3,1,1,2 etc

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df=  pd.read_csv("breast-cancer-wisconsin.data.txt")

# missing data denoted by '?'
# inplace modifies data frame right away
# -99999 is because, most algorithm recognize it as an outlier 
# and treat it as opposed to dumping it treats it as a outlier
df.replace('?', -99999, inplace = True)

# now filter out useless data such as ID 
# ID doesn't have any implication on whether tumor is benign or malignant

# axis = 1, meaning drop the column / you could also use dropna()
df.drop(["id"], 1, inplace = True)

# define x and y
# features are everything except for class column
x = np.array(df.drop(["class"], 1))
# label is the class
y = np.array(df["class"])

# this separates data into training and test sets
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2) # 20%

# classifier
# neighbors.KNeighborsClassifier is a Class
clf = neighbors.KNeighborsClassifier()
# fit is method
clf.fit(x_train, y_train)

# we can test immediately
accuracy = clf.score(x_test, y_test)
print(accuracy)
# accuracy of the classifier
# 0.96 == 96% huge accuracy!

# if you uncomment out excluding the "ID" column
# df.drop(["id"], 1, inplace = True) 
# you get output of 0.56 == 56%

# prediction
# this is the new data point we want to predict using kNN
example_measures = np.array([4,2,1,1,1,2,3,2,1])

# if you get a value error about reshape 'deprecated error'
# pandas reshape : series reshape | now use .values.reshape()
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measure)
print(prediction)
# [2]

# for 2 samples?
example_measures1 = np.array([[4,2,1,1,1,3,2,1,1],[4,2,1,2,2,2,3,2,1]])
example_measures1 = example_measures1.reshape(2, -1)
 
# run the code and get the result
# for any number of samples you don't need to hardcode the reshape
# you can do this
example_measures1 = np.array([[4,2,1,1,1,3,2,1,1],[4,2,1,2,2,2,3,2,1]])
example_measures1 = example_measures1.reshape(len(example_measures1), -1)

"""
	1. first use the data split it into training and test set
	   test if the classifier from sklearn is accurate using training data
	2. if accurate, import np.array(["your own data"]) and use classifier to fit
	3. does it classify to breast cancer? etc?
	4. now build your own machine learning algorithm model and test the data
	5. do they separate nicely? how does it compare to the original model from sklearn?
"""

# -------------------------------------------------------------------------------

# p15. Euclidean Distance

"""
	What is Euclidean Distance?
	
	sqrt(sum(qi - pi)**2)
	
	q = (1,3)
	p = (2,5)

	Euclidean Distance: 

	# since we have 2 dimensions
	sqrt( (1 - 2)^2 + (3 - 5)^2 )
"""
from math import sqrt

plot1 = [1,3]
plot2 = [2,5]

# 2-dimension. Remember it is the sum
euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot[1] - plot2[1])**2)
print(euclidean_distance)
# 2.23606797749979 etc 

# now we know how to calculate euclidean distance we need some more frameworks
# that will take in data set and use kNN to classify a data point!

# -------------------------------------------------------------------------------

# creating our kNN algorithm p16.

import numpy as np
import matplotlib.pyplot as plt
import warnings # warning for users when they put weird values for k
from matplolib import style
from collections import Counter # votes
from math import sqrt
style.use("fivethirtyeight")

# 2 classes and their features
dataset = {"k": [[1,2],[2,3],[3,1]], "r": [[6,5],[7,7],[8,6]]}

""" feature is some information about your data that the learning algorithm 
	uses to make predictions. For example, using size of house (a feature) to 
	predict it's selling price. """

new_features = [5,7] 
# which one does new_feature fit into?

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1], s = 100, color = i)

# one liner for loop
[[plt.scatter(ii[0],ii[1], s = 100, color = i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1])
plt.show()

# visually we think the new_feature belongs to the class 2

""" kNN algorithm definition """

# we need training data, prediction data, k value
def k_nearest_neighbors(data, predict, k = 3):

	if len(data) >= k:
		warnings.warn("K is set to a value less than total voting groups! idiot!")

	knnalogs 
	return vote_result

# -------------------------------------------------------------------------------

# writing our own kNN in code p17.

import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style
from collections import Counter
style.use("fivethirtyeight")

dataset = {"k" : [[1,3] ,[2,3] ,[3,1]], "r": [[6,5], [7,7], [8,6]]}
new_features = [5,7]

[[plt.scatter(ii[0],ii[1], s = 100, color = i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1])
plt.show()

def k_nearest_negihbors(data, predict, k = 3):

	if len(data) >= k:
		warnings.warn("K is set to a value less than total voting groups!")

	distances = []
	for group in data:
		for features in data[group]:
			# euclidean distance calculated like this but this is only for 2 dimensional array
			# euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)	
			# to work out any > 2 dimensional arrays use numpy
			# euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
			# but there is more simpler version: np.linalg.norm() which is a euclidean distance
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]

	# same as
	# for i in sorted(distnaces)[:k]:
	# 	i[1]
	print(Counter(votes).most_common(1))
	vote_results = Counter(votes).most_common(1)[0][0]

	return vote_result

# kNN run the algorithm!
results = k_nearest_negihbors(dataset, new_features, k = 3)
print(result)

# outputs list of tuples
# [("r", 3)]
# r

# -------------------------------------------------------------------------------

# kNN algorithm testing on real world data p18.
# compare accuracy to scikit learn accuracy?
# use the breast cancer patient?

# small example of converting values in data frame
by_column = [df[x].values.tolist() for x in df.columns]

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random # we shuffling the data coz we not using scikit learn

def k_nearest_neighbors(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    	
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    
    vote_results = Counter(votes).most_common(1)[0][0]
    return vote_results

df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace("?", -99999, inplace = True) # this is a significant outliers
df.drop(["id"], 1, inplace = True) # remove useless column
full_data = df.astype(float).values.tolist() # to allow all the values to a float

# shuffle the data now since we converted to a list of list above
print(full_data[:5])
random.shuffle(full_data)
print(20*"#")
print(full_data[:5])

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

# everything up to the last 20% of data
train_data = full_data[:-int(test_size*len(full_data))]
# last 20% of the data
test_data = full_data[-int(test_size*len(full_data)):]

# populate dictionary
for i in train_data:
	train_set[i[-1]].append(i[:-1]) # last value

for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# precict value = test_set
# dataset = train_data
for group in test_set:
	for data in test_set[group]:
		vote = k_nearest_neighbors(train_set, data, k = 5)

		# this shows the result of prediction
		print(vote)

		if group == vote:
			correct += 1

		total += 1

print("Accuracy:", correct/total)
# len(dataset) = 50 and if out of 50, 47 got correct
# if correct we added += 1, then the accuracy is 0.97
# outputs accuracy: 0.97

# Compare this with the scikit-learn algorithms

# -------------------------------------------------------------------------------

# Final thoughts on kNN p19.
# if increased k would accuracy increase? for e.g. k = 25?

# training set and test set
# given some data called "training set", a model is built
# This model generally will try to predict one variable based on all the others

""" We're going to cover a few final thoughts on the K Nearest Neighbors algorithm here, including the value for 
K, confidence, speed, and the pros and cons of the algorithm now that we 
understand more about how it works. """

# confidence? vs Accuracy?

"""
  	kNN can be threaded. Works both linear and non-linear data
  	linear regression only for linear data? 

"""
# -------------------------------------------------------------------------------
# END OF Classification W/ K Nearest Neighbors 


# SVM
# Support Vector Machine (SVM) Part 20.
# Supervised machine learning classifier

# SVM does binary classification - separate into two groups
# denoted by + / -

""" 
Objective of SVM is to identify the best separating hyperplane
Distance between the associated data and the hyperplane is greatest at the hyperplane

How do we get to the best Hyperplane?
Once you have best hyperplane, you have a + data? or - data?
SVM: Binary Classifier and against the linear data 


"""
# code from kNN
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df=  pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace = True)

df.drop(["id"], 1, inplace = True)
x = np.array(df.drop(["class"], 1))
y = np.array(df["class"])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2) # 20%

# support vector machine
clf = svm.SVC()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

# example_measures = np.array([4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1])
# example_measures = example_measures.reshape(len(example_measures), -1)
# prediction = clf.predict(example_measure)
# print(prediction)

# -------------------------------------------------------------------------------

# Understanding Vectors - SVM p21.


# breaking down of the support vector machine
"""
	theoretical background
"""
# -------------------------------------------------------------------------------

# Support Vector Assertion - p22.

"""
	what is operation?

"""
# -------------------------------------------------------------------------------

# SVM p23.





