

import math
import numpy as np
import random


#helper function to bucket continuous variables into x bins
#last bucket may be cut short
def toCategorical(data, numClasses):
	length = len(data)
	categoricalData = []
	currentClass = 0	
	justCategorical = []
	bucketLength = math.ceil(length/numClasses)

	indexData = [(index, item) for index, item in enumerate(data)]
	indexData.sort(key = lambda x: x[1])
	
	for i in range(length):
		categoricalData.append([currentClass, indexData[i][0]])
		if (i + 1) % bucketLength == 0:
			currentClass = currentClass + 1
	categoricalData.sort(key = lambda x: x[1])

	for i in range(length):
		justCategorical.append(categoricalData[i][0])
	return justCategorical


#returns the most common label of the passed data
def mostCommonLabel(data):
	return max(set(list(data)), key = list(data).count)


#is the object which holds the user facing functionality: fit() and predict()
class Dtree:
	def __init__(self, maxDepth = None):
		self.root = None
		self.maxDepth = maxDepth


	def fit(self, data, labels):
		length = len(labels)
		self.root = Node(nodeType = "root", maxDepth = self.maxDepth, nodeX = data, nodeY = labels, depth = 1)
		self.root.buildTree()


	def predict(self, data):
		predValues = []
		for i in data:
			value = self.root.nodePred(i)
			if value != None:
				predValues.append(value)
		return(predValues)


#nodes calculate how to split data when training and hold that information,
#allowing for nodes to split data while making predictions
class Node:
	def __init__(self, nodeX, nodeY, maxDepth, depth, nodeType = None, splittingFeature = None, splittingFeatureClass = None, leafReturn = None):
		self.left = None
		self.right = None
		self.nodeX = nodeX
		self.nodeY = nodeY
		self.maxDepth = maxDepth		
		self.depth = depth
		self.nodeType = nodeType
		self.splittingFeature = splittingFeature
		self.splittingFeatureClass = splittingFeatureClass
		self.leafReturn = leafReturn

	#follow nodes down to a leaf, return the class predicted by that leaf
	def nodePred(self, data):
		if self.nodeType == "leaf":
			return self.leafReturn
		elif (self.nodeType == "root") or (self.nodeType == "decision"):
			if data[self.splittingFeature] == self.splittingFeatureClass:
				return(self.left.nodePred(data))
			else:
				return(self.right.nodePred(data))


	#calculate gini for a given feature
	#returns the average gini for a given feature
	def gini(self, feature, labels):
		length = len(labels)	
		buildCount = 0
		gini = 0
		bestFeature = None
		bestGini = 2
		featureClasses, fcounts = np.unique(feature, return_counts = True)
		labelClasses, lcounts = np.unique(labels, return_counts = True)

		featureIndex = 0	
		for featureClass in featureClasses:
			classGini = 0
			for labelClass in labelClasses:
				for count in range(length):
					if (str(feature[count]) == str(featureClass)) & (str(labels[count]) == str(labelClass)):
						buildCount = buildCount + 1
				if buildCount > 0:
					classGini = classGini + ((buildCount/fcounts[featureIndex])**2)
					buildCount = 0
			if 1-classGini < bestGini:
				bestGini = classGini
				bestFeature = featureClass

			gini = gini + ((fcounts[featureIndex] / length) * (1 - classGini))
			featureIndex = featureIndex + 1

		return(gini, bestFeature)


	#calculate how to split our data based on each features Gini impurity
	def calcSplit(self):
		numCols = self.nodeX.shape[1]
		bestGini = 2
		bestAttribute = -1
		currGini = 2
		splitOn = None
		allSameGini = False
		allGini = []

		for col in range(numCols):
			currGini = self.gini(list(self.nodeX[:,col]), list(self.nodeY))
			allGini.append(currGini[0])
			if currGini[0] < bestGini:
				bestAttribute = col
				bestGini = currGini[0]
				splitOn = currGini[1]

		if len(set(allGini)) == 1:
			allSameGini = True
		return bestAttribute, splitOn, allSameGini


	#recursively create nodes until we reach a leaf
	def buildTree(self):
		#if we are at max depth, return the most common label
		if self.depth == self.maxDepth:
			self.nodeType = "leaf"
			self.leafReturn = mostCommonLabel(self.nodeY)
			return()

		#if every label is the same, return the label
		if len(set(self.nodeY)) == 1:
			self.nodeType = "leaf"
			self.leafReturn = self.nodeY[0]
			return()

		#not a leaf, calc gini, split data, recurse
		splitInfo = self.calcSplit()

		if splitInfo[2] == True:
			self.nodeType = "leaf"
			self.leafReturn = mostCommonLabel(self.nodeY)
			return()				

		self.nodeType = "decision"
		self.splittingFeature = splitInfo[0]
		self.splittingFeatureClass = splitInfo[1]
		leftX = self.nodeX[np.where(self.nodeX[:,splitInfo[0]] == splitInfo[1])]
		leftY = self.nodeY[np.where(self.nodeX[:,splitInfo[0]] == splitInfo[1])]
		rightX = self.nodeX[np.where(self.nodeX[:,splitInfo[0]] != splitInfo[1])]
		rightY = self.nodeY[np.where(self.nodeX[:,splitInfo[0]] != splitInfo[1])]
		self.left = Node(nodeX = leftX, nodeY = leftY, depth = self.depth + 1, maxDepth = self.maxDepth)
		self.right = Node(nodeX = rightX, nodeY = rightY, depth = self.depth + 1, maxDepth = self.maxDepth)
		self.left.buildTree()
		self.right.buildTree()


#uses above decision tree to implement a random forest
class RandomForest:
	def __init__(self, nTrees = 50, maxDepth = None, bootstrap = True):
		self.nTrees = nTrees
		self.maxDepth = maxDepth
		self.bootstrap = bootstrap
		self.treeList = None

	#train the random forest on provided data,targets
	def fit(self, data, labels):
		random.seed(a = 10)
		numFeatures = data.shape[1]
		numObservations = data.shape[0]
		featuresPerTree = math.ceil(math.sqrt(numFeatures))
		self.treeList = [Dtree(maxDepth = self.maxDepth) for i in range(self.nTrees)]

		for tree in self.treeList:
			#select a random subset of features to build our tree using			
			usedFeatures = random.sample(range(numFeatures), featuresPerTree)
			usedData = np.zeros(data.shape)
			for i in range(numFeatures):
				if i in usedFeatures:
					usedData[:,i] = data[:,i]

			#perform bootstrapping on our data
			if self.bootstrap == True:
				bootstrapSample = random.choices(range(numObservations),k=numObservations)
				bootstrapData = usedData[bootstrapSample,:]
				bootstrapLabels = labels[bootstrapSample]
				#if bootstrapped, fit on bootstrapped data
				tree.fit(bootstrapData, bootstrapLabels)

			#if not bootstrapped, fit without bootstrapped data
			elif self.bootstrap == False:
				tree.fit(usedData, labels)

				
	#use the most common prediction from all of our trees as the true prediction
	def predict(self, data):
		finalPredictions = []
		allTreePredictions = []
		for tree in self.treeList:
			allTreePredictions.append(tree.predict(data))
		for i in np.array(allTreePredictions).T:
			finalPredictions.append(mostCommonLabel(i))
		return(finalPredictions)



