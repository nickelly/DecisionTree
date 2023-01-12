

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import tree
import dtree


iris = load_iris()

testData = iris.data
testLabels = iris.target
for i in range(testData.shape[1]):
	testData[:,i] = dtree.toCategorical(testData[:,i], 5)

xtrain, xtest, ytrain, ytest = train_test_split(testData, testLabels, test_size=0.33, random_state=10)


skTree = tree.DecisionTreeClassifier()
testTree = dtree.Dtree()

skTree = skTree.fit(xtrain, ytrain)
testTree.fit(np.array(xtrain), ytrain)

skPred = skTree.predict(xtest)
print("Sklearn decision tree accuracy score: ", accuracy_score(ytest, skPred))
pred = testTree.predict(np.array(xtest))
print("Custom decision tree accuracy: ", accuracy_score(ytest, pred))

