

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/4classdata.csv')

"""##**Applying different models to data**"""

from matplotlib import pyplot as plt

def correlation_matrix(data, filename='output'):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.shape[1]), data.columns, fontsize=10, rotation=90)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.show()
    
correlation_matrix(data)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = data.dropna()
X = data.iloc[:, 0:-1]
X_norm = scaler.fit_transform(X)
#X_norm = scaler.transform(X)
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 0)

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE().fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig, img = plt.subplots(1,1,figsize=(15,5),sharex='col', sharey='row')
  ax11 = img.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("Perplexity 30")
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE(n_components=3).fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig = plt.figure(figsize = (10, 7))
  ax = plt.axes(projection ="3d")
  ax11 = ax.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1],tsneDatasetOneTwoDimPerp_30[:,2], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("Perplexity 30")
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(solver='lbfgs', max_iter=4000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

print("Using cross validation")

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'linear', random_state = 42, gamma = 'scale')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'poly', degree = 2,random_state = 42, gamma = 'scale')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'sigmoid', random_state = 42, gamma = 'scale')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)
acc = model.score(x_test,y_test)
acc1 = model.score(x_train,y_train)
print('training accuracy',acc1)
print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

model = LogisticRegression(solver='lbfgs', max_iter=4000)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred1 = cross_val_predict(model, X_norm, y, cv=cv)

model = SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred2 = cross_val_predict(model, X_norm, y, cv=cv)


model = tree.DecisionTreeClassifier(criterion='entropy')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred3 = cross_val_predict(model, X_norm, y, cv=cv)


model = KNeighborsClassifier(n_neighbors=7)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred4 = cross_val_predict(model, X_norm, y, cv=cv)

from sklearn.ensemble import VotingClassifier
 

def voting():
	
	models = list()
	models.append(('log', LogisticRegression(solver='lbfgs', max_iter=4000)))
	models.append(('svm', SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')))
	models.append(('dt', tree.DecisionTreeClassifier(criterion='entropy')))
	models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
	
	
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

ensemble = voting()
ensemble.fit(X_norm, y)

cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
scores = cross_val_score(ensemble, X_norm, y, cv=cv, scoring='accuracy')
print('Using Ensemble',scores.mean())

yp = ensemble.predict(x_test)
print(accuracy_score(y_test,yp))
yp = ensemble.predict(x_train)
print(accuracy_score(y_train,yp))

data.shape

"""#**5 second dataset**"""

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_5s_all_genres.csv')
data_test = pd.read_csv('/content/drive/My Drive/ML_Project/data_5s_test_all_genres.csv')

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_5s_all_genres.csv')
data_test = pd.read_csv('/content/drive/My Drive/ML_Project/data_5s_test_all_genres.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = data.dropna()
data_test = data_test.dropna()
X = data.iloc[:, 0:-1]
X_norm = scaler.fit_transform(X)
#X_norm = scaler.transform(X)
y = data.iloc[:, -1]
x_test = data_test.iloc[:, 0:-1]
x_test = scaler.fit_transform(x_test)
y_test = data_test.iloc[:, -1]
#x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 0)

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE().fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig, img = plt.subplots(1,1,figsize=(15,5),sharex='col', sharey='row')
  ax11 = img.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("TSNE on 5-sec length samples")
  plt.xlabel("Dimension 1")
  plt.ylabel("Dimension 2")
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE(n_components=3).fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig = plt.figure(figsize = (10, 7))
  ax = plt.axes(projection ="3d")
  ax11 = ax.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1],tsneDatasetOneTwoDimPerp_30[:,2], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("TSNE on 5-sec length samples")
  plt.xlabel("Dimension 1")
  plt.ylabel("Dimension 2")
  plt.zlabel("Dimension 3")
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(solver='lbfgs', max_iter=4000)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

print("Using cross validation")

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'linear', random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'poly', degree = 2,random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 5)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='gini')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred1 = cross_val_predict(model, X_norm, y, cv=cv)

model = SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred2 = cross_val_predict(model, X_norm, y, cv=cv)


model = tree.DecisionTreeClassifier(criterion='entropy')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred3 = cross_val_predict(model, X_norm, y, cv=cv)


model = KNeighborsClassifier(n_neighbors=7)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred4 = cross_val_predict(model, X_norm, y, cv=cv)

from sklearn.ensemble import VotingClassifier
 

def voting():
	
	models = list()
	models.append(('log', LogisticRegression(solver='lbfgs', max_iter=4000)))
	models.append(('svm', SVC(kernel = 'linear', random_state = 42, gamma = 'scale')))
	models.append(('dt', tree.DecisionTreeClassifier(criterion='entropy')))
	models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
	
	
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

ensemble = voting()
#ensemble.fit(X_norm, y)

cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
scores = cross_val_score(ensemble, X_norm, y, cv=cv, scoring='accuracy')
print('Using Ensemble',scores.mean())

import numpy as np
import random
from keras.layers import Conv2D, BatchNormalization, Dense, MaxPool2D, Input, ZeroPadding2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import np_utils
import pickle
from keras.optimizers import SGD
from keras.models import load_model

model = Sequential()
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(x_train,y_train,batch_size=100,epochs=25)

scores = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

"""#**10 second dataset**"""

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_10s_all_genres.csv')
data_test = pd.read_csv('/content/drive/My Drive/ML_Project/data_10s_test_all_genres.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = data.dropna()
data_test = data_test.dropna()
X = data.iloc[:, 0:-1]
X_norm = scaler.fit_transform(X)
#X_norm = scaler.transform(X)
y = data.iloc[:, -1]
x_test = data_test.iloc[:, 0:-1]
x_test = scaler.fit_transform(x_test)
y_test = data_test.iloc[:, -1]
#x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 0)

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE().fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig, img = plt.subplots(1,1,figsize=(15,5),sharex='col', sharey='row')
  ax11 = img.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("TSNE on 10-sec length samples")
  plt.xlabel("Dimension 1")
  plt.ylabel("Dimension 2")
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE(n_components=3).fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig = plt.figure(figsize = (10, 7))
  ax = plt.axes(projection ="3d")
  ax11 = ax.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1],tsneDatasetOneTwoDimPerp_30[:,2], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("TSNE on 10-sec length samples")
  ax.set_xlabel("Dimension 1")
  ax.set_ylabel("Dimension 2")
  ax.set_zlabel("Dimension 2")
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(solver='lbfgs', max_iter=4000)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

print("Using cross validation")

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'linear', random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'poly', degree = 2,random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 5)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='gini')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred1 = cross_val_predict(model, X_norm, y, cv=cv)

model = SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred2 = cross_val_predict(model, X_norm, y, cv=cv)


model = tree.DecisionTreeClassifier(criterion='entropy')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred3 = cross_val_predict(model, X_norm, y, cv=cv)


model = KNeighborsClassifier(n_neighbors=7)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred4 = cross_val_predict(model, X_norm, y, cv=cv)

from sklearn.ensemble import VotingClassifier
 

def voting():
	
	models = list()
	models.append(('log', LogisticRegression(solver='lbfgs', max_iter=4000)))
	models.append(('svm', SVC(kernel = 'linear', random_state = 42, gamma = 'scale')))
	models.append(('dt', tree.DecisionTreeClassifier(criterion='entropy')))
	models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
	
	
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

ensemble = voting()
#ensemble.fit(X_norm, y)

cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
scores = cross_val_score(ensemble, X_norm, y, cv=cv, scoring='accuracy')
print('Using Ensemble',scores.mean())

"""#**20 second dataset**"""

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_20s_all_genres.csv')
data_test = pd.read_csv('/content/drive/My Drive/ML_Project/data_20s_test_all_genres.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = data.dropna()
data_test = data_test.dropna()
X = data.iloc[:, 0:-1]
X_norm = scaler.fit_transform(X)
#X_norm = scaler.transform(X)
y = data.iloc[:, -1]
x_test = data_test.iloc[:, 0:-1]
x_test = scaler.fit_transform(x_test)
y_test = data_test.iloc[:, -1]
#x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 0)

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE().fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig, img = plt.subplots(1,1,figsize=(15,5),sharex='col', sharey='row')
  ax11 = img.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("TSNE on 20-sec length samples")
  plt.xlabel("Dimension 1")
  plt.ylabel("Dimension 2")
  
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

def tsne2D(): # this function is used to apply tsne on dataset_1 to reduce it to two dimensions

# calling tsne for perplexities 30 and 40
  tsneDatasetOneTwoDimPerp_30 = TSNE(n_components=3).fit_transform(X_norm)
  #tsneDatasetOneTwoDimPerp_40 = TSNE(perplexity = 40).fit_transform(x_subset)

# plotting the tsne data 
  fig = plt.figure(figsize = (10, 7))
  ax = plt.axes(projection ="3d")
  ax11 = ax.scatter(tsneDatasetOneTwoDimPerp_30[:,0],tsneDatasetOneTwoDimPerp_30[:,1],tsneDatasetOneTwoDimPerp_30[:,2], c = y)
  #ax12 = img[1].scatter(tsneDatasetOneTwoDimPerp_40[:,0],tsneDatasetOneTwoDimPerp_40[:,1], c = y_subset)
  
  cbar = plt.colorbar(ax11)
  #ticks = [0,1,2,3,4,5,6,7,8,9]
  #cbar.set_ticks(ticks)
  fig.suptitle("TSNE on 20-sec length samples")
  ax.set_xlabel("Dimension 1")
  ax.set_ylabel("Dimension 2")
  ax.set_zlabel("Dimension 3")
  #cbar.set_ticklabels(['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9'])
  plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne2D()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(solver='lbfgs', max_iter=4000)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

print("Using cross validation")

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'rbf', random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'linear', random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.svm import SVC

model = SVC(kernel = 'poly', degree = 2,random_state = 42, gamma = 'scale')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 5)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='gini')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# acc = model.score(x_test,y_test)
# acc1 = model.score(x_train,y_train)
# print('training accuracy',acc1)
# print('testing accuracy',acc)

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold

def cross_validation(clf, X, y):
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    scores = cross_val_score(model, X_norm, y, cv=cv, scoring='accuracy')
    # ypred = cross_val_predict(clf, X, y, cv=cv)
    # accuracy = accuracy_score(y, ypred)
    return scores.mean()

acc = cross_validation(model,X_norm,y)
print(acc)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred1 = cross_val_predict(model, X_norm, y, cv=cv)

model = SVC(kernel = 'linear', random_state = 42, gamma = 'scale')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred2 = cross_val_predict(model, X_norm, y, cv=cv)


model = tree.DecisionTreeClassifier(criterion='entropy')
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred3 = cross_val_predict(model, X_norm, y, cv=cv)


model = KNeighborsClassifier(n_neighbors=7)
cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
y_pred4 = cross_val_predict(model, X_norm, y, cv=cv)

from sklearn.ensemble import VotingClassifier
 

def voting():
	
	models = list()
	models.append(('log', LogisticRegression(solver='lbfgs', max_iter=4000)))
	models.append(('svm', SVC(kernel = 'linear', random_state = 42, gamma = 'scale')))
	models.append(('dt', tree.DecisionTreeClassifier(criterion='entropy')))
	models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
	
	
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

ensemble = voting()
#ensemble.fit(X_norm, y)

cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
scores = cross_val_score(ensemble, X_norm, y, cv=cv, scoring='accuracy')
print('Using Ensemble',scores.mean())

"""#**Using Neural Networks on all the above datasets**

##***5 second dataset***
"""

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_5s_all_genres.csv')

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_5s_all_genres.csv')
data_test = pd.read_csv('/content/drive/My Drive/ML_Project/data_5s_test_all_genres.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = data.dropna()
data_test = data_test.dropna()
X = data.iloc[:, 0:-1]
X_norm = scaler.fit_transform(X)
#X_norm = scaler.transform(X)
y = data.iloc[:, -1]
x_test = data_test.iloc[:, 0:-1]
x_test = scaler.fit_transform(x_test)
y_test = data_test.iloc[:, -1]
#x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 0)
from keras.utils import to_categorical
y = to_categorical(y)
y_test = to_categorical(y_test)



import numpy as np
import random
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.utils import np_utils
import pickle
from keras.optimizers import SGD
from keras.models import load_model

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=29))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
sgd = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_norm,y,batch_size=10,epochs=50)

scores = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

"""#10 sec neural"""

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_10s_all_genres.csv')
data_test = pd.read_csv('/content/drive/My Drive/ML_Project/data_10s_test_all_genres.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = data.dropna()
data_test = data_test.dropna()
X = data.iloc[:, 0:-1]
X_norm = scaler.fit_transform(X)
#X_norm = scaler.transform(X)
y = data.iloc[:, -1]
x_test = data_test.iloc[:, 0:-1]
x_test = scaler.fit_transform(x_test)
y_test = data_test.iloc[:, -1]
#x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 0)
from keras.utils import to_categorical
y = to_categorical(y)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=29))
#model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
sgd = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_norm,y,batch_size=10,epochs=150)

scores = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

"""#20 sec neural"""

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/ML_Project/data_20s_all_genres.csv')
data_test = pd.read_csv('/content/drive/My Drive/ML_Project/data_20s_test_all_genres.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = data.dropna()
data_test = data_test.dropna()
X = data.iloc[:, 0:-1]
X_norm = scaler.fit_transform(X)
#X_norm = scaler.transform(X)
y = data.iloc[:, -1]
x_test = data_test.iloc[:, 0:-1]
x_test = scaler.fit_transform(x_test)
y_test = data_test.iloc[:, -1]
#x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2, random_state = 0)
from keras.utils import to_categorical
y = to_categorical(y)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=29))
#model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
sgd = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_norm,y,batch_size=10,epochs=150)

scores = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

