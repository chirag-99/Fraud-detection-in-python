import pandas as pd
import xgboost as xgb

import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn_pandas import DataFrameMapper
#from sklearn_pandas import CategoricalImputer
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV




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

#######
#Create a boolean mask for categorical columns
categorical_feature_mask = train.dtypes == object
# Get list of categorical column names
categorical_columns = train.columns[categorical_feature_mask].tolist()


### using just 10000 rows because of RAM issues 
X, y = train.iloc[:10000,:-1], train.iloc[:10000,-1]

X.info()
X.shape


###### creating dummy variables for categorical columns

df_with_dummies = pd.get_dummies(X,columns =  categorical_columns, sparse=False)

df_with_dummies.shape
##


## re-covnerting date columns to integers
df_with_dummies['device1_date1'] = df_with_dummies['device1_date1'].astype(int)
df_with_dummies['device2_date1'] = df_with_dummies['device2_date1'].astype(int)
df_with_dummies['device2_date2'] = df_with_dummies['device2_date2'].astype(int)

df_with_dummies.info()
###

### XGBoost model

# Create the DMatrix
euc_dmatrix = xgb.DMatrix(data=df_with_dummies, label=y)



# Create the parameter dictionary: params
params = {"objective":'binary:logistic', "max_depth":5}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=euc_dmatrix, params=params, nfold =3,num_boost_round=5, metrics="auc", seed = 123, as_pandas=True)


# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])
