# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 22:24:19 2022

@author: pauli
"""

import pandas as pd
import datetime as dt

df_save = pd.read_csv("guillaume.txt", sep=";")

df=df_save.sample(n=10000)



#df["DateTransaction_houre"] = pd.to_datetime(df["DateTransaction"]).dt.time
df["DateTransaction_day"] = pd.to_datetime(df["DateTransaction"]).dt.date


df_train = df[df["DateTransaction_day"].between(dt.date(2017,2,1), dt.date(2017,8,31))]
df_test = df[df["DateTransaction_day"].between(dt.date(2017,9,1), dt.date(2017,11,30))] 


Xtrain = df_train.drop('FlagImpaye', axis=1)
Xtest = df_test.drop('FlagImpaye', axis=1)
ytrain = df_train["FlagImpaye"]
ytest = df_test["FlagImpaye"]


X = Xtrain.copy()
X=X.drop(['ZIBZIN','CodeDecision', 'IDAvisAutorisationCheque', 'DateTransaction'], axis=1)
X.iloc[0]
X.info()

var_tofloat = ['Montant','ScoringFP1', 'ScoringFP2', 'ScoringFP3', 'TauxImpNb_RB', 'TauxImpNB_CPM', 'DiffDateTr1', 'DiffDateTr2', 'DiffDateTr3', 'CA3TRetMtt', 'CA3TR']
for i in var_tofloat:
    X[i] = X[i].str.replace(",",".")
    X[i] = pd.to_numeric(X[i], downcast="float")

var_toint = ['D2CB', 'Heure', 'EcartNumCheq', 'VerifianceCPT1', 'VerifianceCPT2', 'VerifianceCPT3', 'NbrMagasin3J']
for i in var_toint:
    X[i] =X[i].astype(int)
    
date_toint = ['DateTransaction_day']
for i in date_toint:
    X[i] = X[i].astype(str)
    X[i] = X[i].str.replace("-","")
    X[i] = X[i].astype(int)

X.info()


from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=1)
ytrain = ytrain.astype(int)
X = X.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
smote_Xtrain, smote_ytrain = sm.fit_resample(X, ytrain)
smote = pd.concat([smote_Xtrain, smote_ytrain], axis = 1)


smote_Xtrain.to_csv("C:/Users/pauli/Documents/M2/fouille de données/projet/smote_Xtrain.csv", index=False)
smote_ytrain.to_csv("C:/Users/pauli/Documents/M2/fouille de données/projet/smote_ytrain.csv", index=False)







