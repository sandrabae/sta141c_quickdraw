###Binary Classification with k-Nearest Neighbors model

#Imports

import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#load the data
cat = np.load("data/cat.npy")
sheep = np.load("data/sheep.npy")

# add a column with labels, 0=cat, 1=sheep
cat = np.c_[cat, np.zeros(len(cat))]
sheep = np.c_[sheep, np.ones(len(sheep))]

# Create the matrices for scikit-learn (5'000 cat and sheep images each):
# merge the cat and sheep arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
X = np.concatenate((cat[:5000,:-1], sheep[:5000,:-1]), axis=0).astype('float32') # all columns but the last
y = np.concatenate((cat[:5000,-1], sheep[:5000,-1]), axis=0).astype('float32') # the last column

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=0, shuffle=True)

start = time.time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

print("Under a binary classification paradigm, the K-Nearest Neighbors",end="")
print(" algorithm can classify images at", neigh.score(X_test, y_test)*100, end="")
print("% accuracy in", time.time()-start, "seconds.")

