from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from scr.ReadData import ReadData
from io import StringIO
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from pydot import graph_from_dot_data
readData =ReadData()
#Reading the data
X,Y = readData.readVideoDataHRV()
scaler = StandardScaler()
X = scaler.fit_transform(X)
# splitting data into training and test set for independent attributes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, Y, test_size=.3,
                                                   random_state=22)
X_train.shape,X_test.shape
#developing a model
clf_pruned = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_pruned.fit(X_train, y_train)
y_pred=clf_pruned.predict(X_test)
print(accuracy_score(y_test,y_pred))
#visualizing the tree
import io

# from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import graphviz
# xvar = X
# feature_cols = xvar.columns
# dot_data = StringIO()
# export_graphviz(clf_pruned, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_cols,class_names=['1','2','3','4','5','6'])

# (graph, ) = graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())