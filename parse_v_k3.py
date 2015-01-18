import json
import glob
import sklearn
import numpy
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import random_projection
from sklearn import cross_validation
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import neighbors
from sklearn.linear_model import Perceptron
from sklearn import svm
from collections import Counter
from sklearn import metrics

#categories = ['others']
#stories = sklearn.datasets.load_files('stories', categories = categories)

def results(clf, X, Y):
	X = X.toarray()
	clf.fit(X, Y)
	predicted = clf.predict(X)
	print numpy.mean(predicted == Y)
	print(metrics.classification_report(Y, predicted))

def results_test(clf, X_train, X_test, Y_train,Y_test):
	X_train = X_train.toarray()
	X_test = X_test.toarray()
	clf.fit(X_train,Y_train)
	predicted = clf.predict(X_test)
	print numpy.mean(predicted == Y_test)
	print(metrics.classification_report(Y_test, predicted))

def run_classifiers(clf, stories_counts_train,stories_counts_test,stories_counts_bin_train,stories_counts_bin_test,stories_tfidf_train,stories_tfidf_test,Y_train,Y_test):
	scores = cross_validation.cross_val_score(clf, stories_counts_train.toarray(), Y_train, cv=10)
	print("Accuracy using count_vect: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	#results(clf, stories_counts, stories.target)
	results_test(clf, stories_counts_train, stories_counts_test,Y_train,Y_test)
	
	scores = cross_validation.cross_val_score(clf, stories_counts_bin_train.toarray(), Y_train, cv=10)
	print("Accuracy using binary counts: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	#results(clf, stories_counts_bin, stories.target)
	results_test(clf, stories_counts_bin_train, stories_counts_bin_test,Y_train,Y_test)

	scores = cross_validation.cross_val_score(clf, stories_tfidf_train.toarray(), Y_train, cv=10)
	print("Accuracy using tfidf: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	#results(clf, stories_tfidf, stories.target)
	results_test(clf, stories_tfidf_train, stories_tfidf_test,Y_train,Y_test)



stories = sklearn.datasets.load_files('stories',shuffle = True)
print len(stories.data)
print Counter(stories.target)
print stories.target_names 

p = int(len(stories.data)*0.75)
print p

X_train = stories.data[:p]
Y_train = stories.target[:p]

X_test = stories.data[p:]
Y_test = stories.target[p:]

count_vect = CountVectorizer(max_features=5000,stop_words= 'english')
#stories_counts = count_vect.fit_transform(stories.data)
stories_counts_train = count_vect.fit_transform(X_train)
stories_counts_test = count_vect.transform(X_test)

count_vect_bin = CountVectorizer(binary=True, max_features=5000, stop_words= 'english')
#stories_counts_bin = count_vect_bin.fit_transform(stories.data)
stories_counts_bin_train = count_vect_bin.fit_transform(X_train)
stories_counts_bin_test = count_vect_bin.transform(X_test)

tfidf_vect = TfidfVectorizer(max_features=5000,stop_words= 'english')
#stories_tfidf = tfidf_vect.fit_transform(stories.data)
stories_tfidf_train = tfidf_vect.fit_transform(X_train)
stories_tfidf_test = tfidf_vect.transform(X_test)


print stories_counts_train.shape
print stories_counts_bin_train.shape
print stories_tfidf_train.shape

print Counter(Y_test)




########################## Multinomial Naive Bayes ########################################
clf = MultinomialNB()
print '-------------Naive Bayes-------------'
run_classifiers(clf,stories_counts_train,stories_counts_test,stories_counts_bin_train,stories_counts_bin_test,stories_tfidf_train,stories_tfidf_test,Y_train,Y_test)


# ########################## Decision Tree ########################################
clf = tree.DecisionTreeClassifier()
print '-------------Decision Tree------------'
run_classifiers(clf,stories_counts_train,stories_counts_test,stories_counts_bin_train,stories_counts_bin_test,stories_tfidf_train,stories_tfidf_test,Y_train,Y_test)


# ########################## KNN ########################################
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
print '----------------KNN------------------'
run_classifiers(clf,stories_counts_train,stories_counts_test,stories_counts_bin_train,stories_counts_bin_test,stories_tfidf_train,stories_tfidf_test,Y_train,Y_test)


# ########################## SVM ########################################
clf = svm.SVC(kernel = 'linear')
print '------------------SVM------------------'
run_classifiers(clf,stories_counts_train,stories_counts_test,stories_counts_bin_train,stories_counts_bin_test,stories_tfidf_train,stories_tfidf_test,Y_train,Y_test)


########################## Perceptron ########################################
clf = Perceptron(n_iter=10)
print '----------------Perceptron--------------'
run_classifiers(clf,stories_counts_train,stories_counts_test,stories_counts_bin_train,stories_counts_bin_test,stories_tfidf_train,stories_tfidf_test,Y_train,Y_test)
# coeff = clf.coef_
# features = tfidf_vect.get_feature_names()
# print features
# print features[0]
# print coeff.shape
# for x in coeff:
# 	n=0
# 	for y in x:
# 		print '{} ==> {}'.format(features[n], y)
# 		n+=1
# 	print '-------------------------------'