# -*- coding: utf-8 -*-
"""
author: ChristiaanML
Naive Bayes method with feature reduction PCA. 
"""

import numpy
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB

#load data
url_train = 'data/train.csv'
names = 'id,feat_1,feat_2,feat_3,feat_4,feat_5,feat_6,feat_7,feat_8,feat_9,feat_10,feat_11,feat_12,feat_13,feat_14,feat_15,feat_16,feat_17,feat_18,feat_19,feat_20,feat_21,feat_22,feat_23,feat_24,feat_25,feat_26,feat_27,feat_28,feat_29,feat_30,feat_31,feat_32,feat_33,feat_34,feat_35,feat_36,feat_37,feat_38,feat_39,feat_40,feat_41,feat_42,feat_43,feat_44,feat_45,feat_46,feat_47,feat_48,feat_49,feat_50,feat_51,feat_52,feat_53,feat_54,feat_55,feat_56,feat_57,feat_58,feat_59,feat_60,feat_61,feat_62,feat_63,feat_64,feat_65,feat_66,feat_67,feat_68,feat_69,feat_70,feat_71,feat_72,feat_73,feat_74,feat_75,feat_76,feat_77,feat_78,feat_79,feat_80,feat_81,feat_82,feat_83,feat_84,feat_85,feat_86,feat_87,feat_88,feat_89,feat_90,feat_91,feat_92,feat_93,target'
names = names.split(',')
dataframe = read_csv(url_train,names = names, header = 1)
X = dataframe.values[:,1:-1]
Y = numpy.char.replace(dataframe.values[:,-1].astype(str),'Class_','').astype(numpy.int)

#create pipeline
estimators = []
estimators.append(('bayes',GaussianNB))
model = Pipeline(estimators)

#evaluate pipeline
seed = 7
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


