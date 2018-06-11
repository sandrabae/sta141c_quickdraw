###Total Classification with k-Nearest Neighbors model

#Imports

import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#load the data
cat = np.load("data/cat.npy")
sheep = np.load("data/sheep.npy")
book = np.load("data/book.npy")
basket = np.load("data/basket.npy")


# add a column with labels, 0=cat, 1=sheep, 2=book, 3=basket
cat = np.c_[cat, np.zeros(len(cat))]
sheep = np.c_[sheep, np.ones(len(sheep))]
book = np.c_[book, np.full((len(book)),2.)]
basket = np.c_[basket, np.full((len(basket)),3.)]


# Create the matrices for scikit-learn (5'000 cat and sheep images each):
# merge the cat and sheep arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
X = np.concatenate((cat[:5000,:-1], sheep[:5000,:-1], book[:5000,:-1], basket[:5000,:-1]),
                    axis=0).astype('float32') # all columns but the last
y = np.concatenate((cat[:5000,-1], sheep[:5000,-1], book[:5000,-1], basket[:5000,-1]),
                   axis=0).astype('float32') # the last column

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=0, shuffle=True)

for i in [1,2,3,4,5]:
    start = time.time()
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)

    print(i, ":")
    
    print("With a four category dataset, the K-Nearest Neighbors",end="")
    print(" algorithm can classify images at", neigh.score(X_test, y_test)*100, end="")
    print("% accuracy in", time.time()-start, "seconds.\n")



