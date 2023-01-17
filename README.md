# Decision Tree
A Gini impurity based decision tree classifier implemented in python using numpy.

Also includes a random forest implemented using said decision tree.

### Usage
fit(x, y) function accepts x and y as numpy arrays and trains the dtree/RandomForest object it is called from on the data provided.

predict(xTest) function called on the trained dtree/RandomForest object returns a predicted class for each entry in xTest.

Decision tree accepts a maxDepth = n parameter to stop tree growth at the specified max depth.
Random forest accepts maxDepth = n, nTrees = n, and bootstrap = True/False parameter to control depth and number of trees, and if bootstrapped data is used.

The model is implemented to work with categorical data, the helper function toCategorical(data, numberOfClasses) accepts numeric data and returns categorical data that is the numeric data split into numberOfClasses bins.
### Versions
Built with python v3.7.11 and numpy v1.18.5

Testing done with scikit-learn v1.0.2 
