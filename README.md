# Decision Tree
A Gini impurity based decision tree classifier implemented in python using numpy.

### Usage
fit(x, y) function accepts x and y as numpy arrays and trains the dtree object it is called from on the data provided.

predict(xTest) function called on the trained dtree object returns a predicted class for each entry in xTest.


The model is implemented to work with categorical data, the helper function toCategorical(data, numberOfClasses) accepts numeric data and returns categorical data that is the numeric data split into numberOfClasses bins.
### Versions
Built with python v3.7.11 and numpy v1.18.5

Testing done with scikit-learn v1.0.2 
