#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:09:12 2018


"""
import pandas as pd
import xgboost as xgb

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV




train = pd.read_csv("/Volumes/KINGSTON/euclid/assessment_data.csv")

train.info()
train.shape
train.describe()
train.head()


train['device1_date1'] = pd.to_datetime(train['device1_date1'])
train['device2_date1'] = pd.to_datetime(train['device2_date1'])
train['device2_date2'] = pd.to_datetime(train['device2_date2'])


#train['additional_feature1'].astype(object)
train['additional_feature3'] = train['additional_feature3'].astype(int)
train['additional_feature4'] = train['additional_feature4'].astype(int)
train['additional_feature5'] = train['additional_feature5'].astype(int)
train['device2_bool1'] = train['device2_bool1'].astype(object)
train['classification'] = train['classification'].astype(object)


del train['Unnamed: 0']


# Check number of nulls in each feature column
nulls_per_column = train.isnull().sum()
print(nulls_per_column)

# Fill missing values with 0
train.device2_bool1 = train.device2_bool1.fillna(0)

X, y = train.iloc[:10000,:-1], train.iloc[:10000,-1]

X.info()
X.shape
####### LABEL ENCODER AND ONE HOT ENCODER
# Create a boolean mask for categorical columns
categorical_mask = (X.dtypes == object)

# Get list of categorical column names
categorical_columns = X.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(X[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
X[categorical_columns] = X[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(X[categorical_columns].head(25))
print(X[categorical_columns].describe())
print(X[categorical_columns].info())

\####### OHE

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask,sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(X)



# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)





#######
#Create a boolean mask for categorical columns
categorical_feature_mask = train.dtypes == object
# Get list of categorical column names
categorical_columns = train.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = train.columns[~categorical_feature_mask].tolist()


#######3


# Convert df into a dictionary: df_dict
df_dict = X.to_dict('records')

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)


######

df_with_dummies = pd.get_dummies(X,columns =  categorical_columns, sparse=False)

df_with_dummies.shape
##






# Create full pipeline
pipeline = Pipeline([
                     #("st_scaler", StandardScaler()),
                     #("dictifier", Dictifier()),
                     #("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])




# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=3)

# Print avg. AUC 
print("3-fold AUC: ", np.mean(cross_val_scores))



###
X_train, X_test, y_train, y_test= train_test_split(df_with_dummies, y,test_size=0.2, random_state=123)


gbm = xgb.XGBClassifier(max_depth=3, n_estimators=2, learning_rate=0.05).fit(X_train, y_train)
predictions = gbm.predict(X_test)

df_with_dummies['device1_date1'] = df_with_dummies['device1_date1'].astype(int)
df_with_dummies['device2_date1'] = df_with_dummies['device2_date1'].astype(int)
df_with_dummies['device2_date2'] = df_with_dummies['device2_date2'].astype(int)

df_with_dummies.info()
###


accuracy = float(np.sum(predictions==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))


train.info()

# Create the parameter dictionary: params
params = {"objective":'binary:logistic', "max_depth":5}


# Create the DMatrix
euc_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=euc_dmatrix, params=params, nfold =3,num_boost_round=5, metrics="auc", seed = 123, as_pandas=True)


# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])
