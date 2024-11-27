#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libs
import os
import numpy as np # Linear Algebra
import pandas as pd # Data Manipulation
pd.set_option('MAX_ROWS', None) # Setting pandas to display a N number of columns
from collections import Counter # Data Manipulation
import seaborn as sns # Data Viz
import matplotlib.pyplot as plt # Data Viz
from sklearn import tree # Modelling a tree
from sklearn.impute import SimpleImputer # Perform Imputation
from imblearn.over_sampling import SMOTE # Perform oversampling
from sklearn.preprocessing import OneHotEncoder # Perform OneHotEnconding
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict # Cross Validation
from sklearn.linear_model import LogisticRegression # Modelling
from sklearn.metrics import classification_report, roc_auc_score,precision_score,recall_score # Evaluating the Model


#warnings
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Collecting the data

# In[ ]:


# Collecting data
df_2019 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')
df_2020 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')
df_2019.head()


# # Problem definition.
# 
# Predict whether a particular flight will be delayed or not. The data refer to flights from January-19 and January-20, so we can use the data to predict flight delays in January for the next period (year-2020).
# 
#  - Binary classification problem.
#  - 21 variables per dataset.
#  - Dataset with flights from Jan-19 and Jan-20.
#  - Variable response is 'ARR_DEL15'
# 
# Variable dictionary:
# 
#     'DAY_OF_MONTH': Day of the month.
#     'DAY_OF_WEEK': Day of the week.
#     'OP_UNIQUE_CARRIER': Unique transport code.
#     'OP_CARRIER_AIRLINE_ID': Unique aviation operator code.
#     'OP_CARRIER': IATA code of the operator.
#     'TAIL_NUM': Tail number.
#     'OP_CARRIER_FL_NUM': Flight number.
#     'ORIGIN_AIRPORT_ID': Origin airport ID.
#     'ORIGIN_AIRPORT_SEQ_ID': Origin airport ID - SEQ.
#     'ORIGIN': Airport of Origin.
#     'DEST_AIRPORT_ID': ID of the destination airport.
#     'DEST_AIRPORT_SEQ_ID': Destination airport ID - SEQ.
#     'DEST': Destination airport.
#     'DEP_TIME': Flight departure time.
#     'DEP_DEL15': Departure delay indicator
#     'DEP_TIME_BLK': block of time (hour) where the match has been postponed.
#     'ARR_TIME': Flight arrival time.
#     'ARR_DEL15': Arrival delay indicator.
#     'CANCELLED': Flight cancellation indicator.
#     'DIVERTED': Indicator if the flight has been diverted.
#     'DISTANCE': Distance between airports.

# # Unifying the bases.
# 
# We will unify the bases of 2019 and 2020 to analyze the data as a whole.

# In[ ]:


#Creating year indicator.
df_2019['year'] = 2019
df_2020['year'] = 2020

#Checking if the bases have the same columns
print(set(df_2020.columns) == set(df_2019.columns))

#Generating the unique base
dataset = pd.concat([df_2019,df_2020])
print(dataset.shape)
print('\n')
dataset.head()


# In[ ]:


data = dataset.drop(['OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID','OP_CARRIER','TAIL_NUM', 'ORIGIN_AIRPORT_ID','ORIGIN_AIRPORT_SEQ_ID','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','Unnamed: 21', 'DEP_DEL15'], axis=1)
data = data.set_index('OP_CARRIER_FL_NUM')
data.head()


# In[ ]:


data.head()


# In[ ]:


#Dataframe summary
pd.DataFrame({'unicos':data.nunique(),
              'missing': data.isna().sum()/data.count(),
              'tipo':data.dtypes})


# In[ ]:


#Missing values
data.dropna(inplace=True)

#Transformation of data types
colunas = ['DAY_OF_WEEK','DAY_OF_MONTH','ARR_DEL15','CANCELLED','DIVERTED']
for col in colunas:
  data[col] = data[col].astype('category') 

#Discretization
data['DISTANCE_cat'] = pd.qcut(data['DISTANCE'], q=4)


# In[ ]:


#Dataframe summary after pre-processing
pd.DataFrame({'unicos':data.nunique(),
              'missing': data.isna().mean()*100,
              'tipo':data.dtypes})


# In[ ]:


#check data
data.head()


# In[ ]:


# Helper function to create ARR_TIME_BLOCK
def arr_time(x):

  if x >= 600 and x <= 659:
    return '0600-0659'
  elif x>=1400 and x<=1459:
    return '1400-1459'
  elif x>=1200 and x<=1259:
    return '1200-1259'
  elif x>=1500 and x<=1559:
    return '1500-1559'
  elif x>=1900 and x<=1959:
    return '1900-1959'
  elif x>=900 and x<=959:
    return '0900-0959'
  elif x>=1000 and x<=1059:
    return  '1000-1059'
  elif x>=2000 and x<=2059:
    return '2000-2059'
  elif x>=1300 and x<=1359:
    return '1300-1359'
  elif x>=1100 and x<=1159:
    return '1100-1159'
  elif x>=800 and x<=859:
    return '0800-0859'
  elif x>=2200 and x<=2259:
    return '2200-2259'
  elif x>=1600 and x<=1659:
    return '1600-1659'
  elif x>=1700 and x<=1759:
    return '1700-1759'
  elif x>=2100 and x<=2159:
    return '2100-2159'
  elif x>=700 and x<=759:
    return '0700-0759'
  elif x>=1800 and x<=1859:
    return '1800-1859'
  elif x>=1 and x<=559:
    return '0001-0559'
  elif x>=2300 and x<=2400:
    return '2300-2400'


# In[ ]:


# We can create ARR_TIME_BLOCK.
data['ARR_TIME'] = data['ARR_TIME'].astype('int')
data['ARR_TIME_BLOCK'] = data['ARR_TIME'].apply(lambda x :arr_time(x))
data.reset_index(inplace=True)
data.head()


# In[ ]:


# Amount of delays within a DEP_TIME_BLK.
count_time_blk = data[['DEP_TIME_BLK','ARR_DEL15']].groupby('DEP_TIME_BLK').sum().sort_values(by='ARR_DEL15',ascending=False)
count_time_blk.reset_index(inplace=True)
count_time_blk.head()
data1 = data.merge(count_time_blk, left_on='DEP_TIME_BLK', right_on='DEP_TIME_BLK') 
data1.rename({'ARR_DEL15_y':'quant_dep_time_blk','ARR_DEL15_x':'ARR_DEL15' }, inplace=True, axis=1)
data1.head()


# In[ ]:


# Number of delays DEP_DEL15 per ORIGIN.
count_later_origin = data[['ORIGIN']].groupby('ORIGIN').sum()
count_later_origin.reset_index(inplace=True)
count_later_origin.head()
data2 = data1.merge(count_later_origin, left_on='ORIGIN', right_on='ORIGIN')
data2.rename({'DEP_DEL15_y':'count_later_origin' }, inplace=True, axis=1)
data2.head() 


# In[ ]:


# Number of delays ARR_DEL15 per DEST.
count_later_dest = data[['DEST','ARR_DEL15']].groupby('DEST').sum().sort_values(by='ARR_DEL15',ascending=False)
count_later_dest.reset_index(inplace=True)
count_later_dest.head()
data3 = data2.merge(count_later_dest, left_on='DEST', right_on='DEST')
data3.rename({'ARR_DEL15_y':'count_later_dest','ARR_DEL15_x':'ARR_DEL15' },inplace=True, axis=1)
data3.head()


# In[ ]:


#Data Preparation
base_final = data3.copy()
base_final.drop(['DEP_TIME','ARR_TIME','OP_CARRIER_FL_NUM'], inplace=True, axis=1)
base_final.set_index('year',inplace=True)


# In[ ]:


# Separate target, numeric and categorical variables 'ORIGIN', 'DEST'

target_final = base_final[['ARR_DEL15']]

cat_vars_final = base_final.select_dtypes(['object','category'])
cat_vars_final = cat_vars_final.loc[:, ['DAY_OF_MONTH', 'DAY_OF_WEEK','DEP_TIME_BLK','CANCELLED',
                            'DIVERTED','DISTANCE_cat','ARR_TIME_BLOCK']]

#One Hot Encoder

enc = OneHotEncoder().fit(cat_vars_final)

cat_vars_ohe_final = enc.transform(cat_vars_final).toarray()
cat_vars_ohe_final = pd.DataFrame(cat_vars_ohe_final, index= cat_vars_final.index, 
                      columns=enc.get_feature_names(cat_vars_final.columns.tolist()))


# In[ ]:


#Logisitc Regression Model


#Dividing into training and test data: 2019 - training, 2020 - testing
target_2019_final = target_final[target_final.index == 2019]
target_2020_final = target_final[target_final.index == 2020]

cat_vars_ohe_2019_final = cat_vars_ohe_final[cat_vars_ohe_final.index == 2019]
cat_vars_ohe_2020_final = cat_vars_ohe_final[cat_vars_ohe_final.index == 2020]


# In[ ]:





# In[ ]:


get_ipython().system('pip install xgboost')


# In[ ]:


from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import numpy as np
import xgboost as xgb


# In[ ]:


# Ensure target variables are integers
target_2019_final = target_2019_final.astype(int)
target_2020_final = target_2020_final.astype(int)


# In[ ]:


cat_vars_ohe_2019_final.head()


# In[ ]:


cat_vars_ohe_2019_final.columns


# In[ ]:


# Fix feature names after one-hot encoding
cat_vars_ohe_2019_final.columns = [
    col.replace("[", "").replace("]", "").replace("<", "").replace(",", "_")
    for col in cat_vars_ohe_2019_final.columns
]

cat_vars_ohe_2020_final.columns = [
    col.replace("[", "").replace("]", "").replace("<", "").replace(",", "_")
    for col in cat_vars_ohe_2020_final.columns
]


# In[ ]:


# Convert to DMatrix format for XGBoost
dtrain = xgb.DMatrix(cat_vars_ohe_2019_final, label=target_2019_final)
dtest = xgb.DMatrix(cat_vars_ohe_2020_final, label=target_2020_final)

# Calculate scale_pos_weight to address class imbalance
scale_pos_weight = len(target_2019_final[target_2019_final == 0]) / len(target_2019_final[target_2019_final == 1])

# Define optimized XGBoost parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.01,  # Lower learning rate for better generalization
    "max_depth": 4,        # Reduce depth to prevent overfitting
    "min_child_weight": 5, # Regularization to avoid overfitting
    "subsample": 0.8,      # Row sampling
    "colsample_bytree": 0.8, # Feature sampling
    "scale_pos_weight": scale_pos_weight,  # Adjust for class imbalance
    "gamma": 1,            # Minimum loss reduction for split
    "reg_lambda": 1,       # L2 regularization
    "seed": 42             # Ensure reproducibility
}


# In[ ]:


# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier

# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 300],
#     'scale_pos_weight': [1, scale_pos_weight]
# }

# grid_search = GridSearchCV(
#     estimator=XGBClassifier(),
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )

# grid_search.fit(cat_vars_ohe_2019_final, target_2019_final)
# best_model = grid_search.best_estimator_
# print(f"Best parameters: {grid_search.best_params_}")


# In[ ]:


# Train the XGBoost model with early stopping
evals_result = {}
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,  # Allow more rounds
    evals=[(dtrain, "train"), (dtest, "validation")],
    early_stopping_rounds=50,  # Stop if validation AUC does not improve
    evals_result=evals_result,
    verbose_eval=True
)


# In[ ]:


# Make predictions
y_pred_prob = xgb_model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)


# In[ ]:


# Evaluate classification performance
print("Classification Report:")
print(classification_report(target_2020_final, y_pred))
print(f"AUC: {roc_auc_score(target_2020_final, y_pred_prob):.4f}")


# In[ ]:


# Find and use the optimal threshold
fpr, tpr, thresholds = roc_curve(target_2020_final, y_pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold}")


# In[ ]:


y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)
print("Classification Report with Optimal Threshold:")
print(classification_report(target_2020_final, y_pred_optimal))
print(f"AUC with Optimal Threshold: {roc_auc_score(target_2020_final, y_pred_prob):.4f}")


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities
y_pred_prob = xgb_model.predict(dtest)

# Compute FPR, TPR, and thresholds
fpr, tpr, thresholds = roc_curve(target_2020_final, y_pred_prob)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"XGBoost ROC Curve (AUC = {roc_auc:.4f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Random Guess")
plt.title("ROC Curve for XGBoost Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# In[ ]:


xgb_model.save_model("xgboost_model.json")
from IPython.display import FileLink

# Generate a download link
FileLink("xgboost_model.json")


# # Load the model in JSON format
# xgb_model_inference = xgb.Booster()
# xgb_model_inference.load_model("xgboost_model.json")
# 
# # Optional: Load from a pickle file
# with open("xgboost_model.pkl", "rb") as f:
#     xgb_model_inference = pickle.load(f)
#     
# # Prepare inference data as DMatrix
# dtest_inference = xgb.DMatrix(cat_vars_ohe_2020_final)  # Use one-hot encoded test data
# 
# # Make predictions
# y_pred_inference = xgb_model_inference.predict(dtest_inference)
# y_pred_binary = (y_pred_inference > 0.5).astype(int)
# 
# # Output results
# print("Predictions:", y_pred_binary)
