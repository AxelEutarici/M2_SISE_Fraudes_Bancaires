# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 09:25:17 2022

@author: pauli
"""

import pandas as pd
Xtrain = pd.read_csv("smote_Xtrain.csv", sep=",")
ytrain = pd.read_csv("smote_ytrain.csv", sep=",")
Xtrain.info()
train = pd.concat([Xtrain, ytrain], axis=1)


#XGBoost/ gradient tree boosting 
from sklearn.ensemble import GradientBoostingClassifier 
param = {"loss":"log_loss","learning_rate":0.1,"n_estimators":100,"min_samples_split":2}
gbc = GradientBoostingClassifier(loss="log_loss", learning_rate=0.1, n_estimators=100, min_samples_split=2)
gbc.fit(Xtrain, ytrain)

#Nearest-Neighbor
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
knc.fit(Xtrain, ytrain)

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
param = {"criterion":"gini","max_depth":None,"min_samples_split":2,"min_samples_leaf":1,"max_features":"sqrt"}
dtc = DecisionTreeClassifier()
dtc = dtc.fit(Xtrain, ytrain)

#Random Forests
from sklearn.ensemble import RandomForestClassifier
param = {"n_estimators":100,"criterion":"gini","max_depth":None,"min_samples_split":2,"min_samples_leaf":1,"max_features":"sqrt","oob_score":False,"warm_start":False,"max_samples":None}
rfc = RandomForestClassifier()
rfc = rfc.fit(Xtrain, ytrain)

#SVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
param = {"kernel":"rbf","degree":3,}
#ici on peut changer le noyaux
svc = make_pipeline(StandardScaler(), SVC(kernel="rbf",degree=3))
svc.fit(Xtrain, ytrain)

#K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(Xtrain)
#bof ca, on ne prend meme pas en compte les y...

#Sampling
###A FORCEMENT FAIRE !!!!!

#LOF
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=2)
lof.fit(train) #train contient X et y 

#Auto-encodeurs
#keskecé ???

#Reseaux de neurones
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier(random_state=1, max_iter=100)
mlpc.fit(Xtrain, ytrain)

#ADL
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
adl = LinearDiscriminantAnalysis()
adl.fit(Xtrain, ytrain)

#ADQ
from sklearn.qda import QDA
qda = QDA()
qda.fit(Xtrain, ytrain)

#Cost-sensitive learning
#On pondère les erreurs 
#Modifier le poids de chaque classe sur le substitue de taux d’erreur 
#Attribuer un poids a chaque entrée de la matrice de confusion (cout a l’échelle de chaque classe) 

#Methodes ensemblistes
#bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
param = {"max_features":0.5,"max_samples" : 0.5}
bagging = BaggingClassifier(KNeighborsClassifier())

#boosting 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
adab = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(adab, Xtrain, ytrain, cv=5)

#regression logistique
from sklearn.linear_model import LogisticRegression
param = {"solver":"saga","penalty":"none","max_iter":100}
logit = LogisticRegression(solver="saga", penalty="none", max_iter=100, random_state=1)
logit.fit(Xtrain, ytrain)


#Metric Learning
from sklearn.metrics import mean_squared_error, recall_score, f1_score, make_scorer,precision_score
#mesure de perg : 
    #Accuracy
    #précision
    #rappel
    #F-mesure = (2TP)/(2TP + FN + FP)
    #courbe ROC (AUC ROC).

