from flask import Flask, jsonify, request

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import plot_tree

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask Server"


def dist_cat(x):
    if x < 368:
        return '(30.999, 368.0]'
    if x < 641:
        return '(368.0, 641.0]'
    if x < 1042:
        return '(641.0, 1042.0]'
    else:
        return '(1042.0, 5095.0]'

def convert_to_feature_vector(day_of_month=0, day_of_week=0, dep_del15=0, dep_time=0, distance_cat=0, arr_time=0):
    """
    Converts more reasonable input features into a one hot feature vector format manually 
    Will require some further preprocessing in order to make this reasonable -- e.g. categorizing departure times into a departure time block
    and sorting distances into a distance time block, etc.
    Parameters:
    - day_of_month (int): Day of the month (1-31)
    - day_of_week (int): Day of the week (1=Monday, 7=Sunday)
    - dep_del15 (float): Departure delay indicator (0.0 or 1.0)
    - dep_time_blk (str): Projected departure time block (e.g., "0600-0659")
    - distance_cat (str): Distance category (e.g., "(30.999, 368.0]") (miles)
    - arr_time_block (str): Projected arrival time block (e.g., "1600-1659")
    
    Returns:
    - pd.DataFrame: One-hot encoded feature vector
    """
    # Initialize all columns to 0
    columns = [
        'DAY_OF_MONTH_1', 'DAY_OF_MONTH_2', 'DAY_OF_MONTH_3', 'DAY_OF_MONTH_4',
        'DAY_OF_MONTH_5', 'DAY_OF_MONTH_6', 'DAY_OF_MONTH_7', 'DAY_OF_MONTH_8',
        'DAY_OF_MONTH_9', 'DAY_OF_MONTH_10', 'DAY_OF_MONTH_11',
        'DAY_OF_MONTH_12', 'DAY_OF_MONTH_13', 'DAY_OF_MONTH_14',
        'DAY_OF_MONTH_15', 'DAY_OF_MONTH_16', 'DAY_OF_MONTH_17',
        'DAY_OF_MONTH_18', 'DAY_OF_MONTH_19', 'DAY_OF_MONTH_20',
        'DAY_OF_MONTH_21', 'DAY_OF_MONTH_22', 'DAY_OF_MONTH_23',
        'DAY_OF_MONTH_24', 'DAY_OF_MONTH_25', 'DAY_OF_MONTH_26',
        'DAY_OF_MONTH_27', 'DAY_OF_MONTH_28', 'DAY_OF_MONTH_29',
        'DAY_OF_MONTH_30', 'DAY_OF_MONTH_31', 'DAY_OF_WEEK_1', 'DAY_OF_WEEK_2',
        'DAY_OF_WEEK_3', 'DAY_OF_WEEK_4', 'DAY_OF_WEEK_5', 'DAY_OF_WEEK_6',
        'DAY_OF_WEEK_7', 'DEP_DEL15_0.0', 'DEP_DEL15_1.0',
        'DEP_TIME_BLK_0001-0559', 'DEP_TIME_BLK_0600-0659',
        'DEP_TIME_BLK_0700-0759', 'DEP_TIME_BLK_0800-0859',
        'DEP_TIME_BLK_0900-0959', 'DEP_TIME_BLK_1000-1059',
        'DEP_TIME_BLK_1100-1159', 'DEP_TIME_BLK_1200-1259',
        'DEP_TIME_BLK_1300-1359', 'DEP_TIME_BLK_1400-1459',
        'DEP_TIME_BLK_1500-1559', 'DEP_TIME_BLK_1600-1659',
        'DEP_TIME_BLK_1700-1759', 'DEP_TIME_BLK_1800-1859',
        'DEP_TIME_BLK_1900-1959', 'DEP_TIME_BLK_2000-2059',
        'DEP_TIME_BLK_2100-2159', 'DEP_TIME_BLK_2200-2259',
        'DEP_TIME_BLK_2300-2359', 'CANCELLED_0.0', 'DIVERTED_0.0',
        'DISTANCE_cat_(30.999, 368.0]', 'DISTANCE_cat_(368.0, 641.0]',
        'DISTANCE_cat_(641.0, 1042.0]', 'DISTANCE_cat_(1042.0, 5095.0]',
        'ARR_TIME_BLOCK_0001-0559', 'ARR_TIME_BLOCK_0600-0659',
        'ARR_TIME_BLOCK_0700-0759', 'ARR_TIME_BLOCK_0800-0859',
        'ARR_TIME_BLOCK_0900-0959', 'ARR_TIME_BLOCK_1000-1059',
        'ARR_TIME_BLOCK_1100-1159', 'ARR_TIME_BLOCK_1200-1259',
        'ARR_TIME_BLOCK_1300-1359', 'ARR_TIME_BLOCK_1400-1459',
        'ARR_TIME_BLOCK_1500-1559', 'ARR_TIME_BLOCK_1600-1659',
        'ARR_TIME_BLOCK_1700-1759', 'ARR_TIME_BLOCK_1800-1859',
        'ARR_TIME_BLOCK_1900-1959', 'ARR_TIME_BLOCK_2000-2059',
        'ARR_TIME_BLOCK_2100-2159', 'ARR_TIME_BLOCK_2200-2259',
        'ARR_TIME_BLOCK_2300-2400'
    ]
    feature_vector = {col: 0.0 for col in columns}
    
    # Set the corresponding one-hot encoded values
    feature_vector[f'DAY_OF_MONTH_{day_of_month}'] = 1.0
    feature_vector[f'DAY_OF_WEEK_{day_of_week}'] = 1.0
    feature_vector[f'DEP_DEL15_{dep_del15}'] = 1.0
    feature_vector[f'DEP_TIME_BLK_{time_block(dep_time)}'] = 1.0
    feature_vector[f'DISTANCE_cat_{distance_cat}'] = 1.0
    feature_vector[f'ARR_TIME_BLOCK_{time_block(arr_time)}'] = 1.0
    
    # Convert to a DataFrame
    return pd.DataFrame([feature_vector])

# Helper function to create ARR_TIME_BLOCK
def time_block(x):

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


@app.route('/api/data', methods=['POST'])
def post_data():
    """
    Flask API endpoint for predicting flight delays using XGBoost model.
    """
    try:
        # Parse input JSON data
        request_data = request.get_json()
        day_of_month = request_data.get("day_of_month", 1)  # Default to 1
        day_of_week = request_data.get("day_of_week", 1)    # Default to Monday
        dep_del15 = request_data.get("dep_del15", 0.0)      # Default to no delay
        dep_time = request_data.get("dep_time", 600)        # Default to 6:00 AM
        distance = request_data.get("distance", 300)        # Default to 300 miles
        arr_time = request_data.get("arr_time", 800)        # Default to 8:00 AM

        # Convert input to feature vector
        distance_cat = dist_cat(distance)
        feature_vector = convert_to_feature_vector(
            day_of_month=day_of_month,
            day_of_week=day_of_week,
            dep_del15=dep_del15,
            dep_time=dep_time,
            distance_cat=distance_cat,
            arr_time=arr_time,
        )

        # Ensure feature vector matches the model's expected column order
        expected_columns = [
        'DAY_OF_MONTH_1', 'DAY_OF_MONTH_2', 'DAY_OF_MONTH_3', 'DAY_OF_MONTH_4',
        'DAY_OF_MONTH_5', 'DAY_OF_MONTH_6', 'DAY_OF_MONTH_7', 'DAY_OF_MONTH_8',
        'DAY_OF_MONTH_9', 'DAY_OF_MONTH_10', 'DAY_OF_MONTH_11',
        'DAY_OF_MONTH_12', 'DAY_OF_MONTH_13', 'DAY_OF_MONTH_14',
        'DAY_OF_MONTH_15', 'DAY_OF_MONTH_16', 'DAY_OF_MONTH_17',
        'DAY_OF_MONTH_18', 'DAY_OF_MONTH_19', 'DAY_OF_MONTH_20',
        'DAY_OF_MONTH_21', 'DAY_OF_MONTH_22', 'DAY_OF_MONTH_23',
        'DAY_OF_MONTH_24', 'DAY_OF_MONTH_25', 'DAY_OF_MONTH_26',
        'DAY_OF_MONTH_27', 'DAY_OF_MONTH_28', 'DAY_OF_MONTH_29',
        'DAY_OF_MONTH_30', 'DAY_OF_MONTH_31', 'DAY_OF_WEEK_1', 'DAY_OF_WEEK_2',
        'DAY_OF_WEEK_3', 'DAY_OF_WEEK_4', 'DAY_OF_WEEK_5', 'DAY_OF_WEEK_6',
        'DAY_OF_WEEK_7', 'DEP_DEL15_0.0', 'DEP_DEL15_1.0',
        'DEP_TIME_BLK_0001-0559', 'DEP_TIME_BLK_0600-0659',
        'DEP_TIME_BLK_0700-0759', 'DEP_TIME_BLK_0800-0859',
        'DEP_TIME_BLK_0900-0959', 'DEP_TIME_BLK_1000-1059',
        'DEP_TIME_BLK_1100-1159', 'DEP_TIME_BLK_1200-1259',
        'DEP_TIME_BLK_1300-1359', 'DEP_TIME_BLK_1400-1459',
        'DEP_TIME_BLK_1500-1559', 'DEP_TIME_BLK_1600-1659',
        'DEP_TIME_BLK_1700-1759', 'DEP_TIME_BLK_1800-1859',
        'DEP_TIME_BLK_1900-1959', 'DEP_TIME_BLK_2000-2059',
        'DEP_TIME_BLK_2100-2159', 'DEP_TIME_BLK_2200-2259',
        'DEP_TIME_BLK_2300-2359', 'CANCELLED_0.0', 'DIVERTED_0.0',
        'DISTANCE_cat_(30.999, 368.0]', 'DISTANCE_cat_(368.0, 641.0]',
        'DISTANCE_cat_(641.0, 1042.0]', 'DISTANCE_cat_(1042.0, 5095.0]',
        'ARR_TIME_BLOCK_0001-0559', 'ARR_TIME_BLOCK_0600-0659',
        'ARR_TIME_BLOCK_0700-0759', 'ARR_TIME_BLOCK_0800-0859',
        'ARR_TIME_BLOCK_0900-0959', 'ARR_TIME_BLOCK_1000-1059',
        'ARR_TIME_BLOCK_1100-1159', 'ARR_TIME_BLOCK_1200-1259',
        'ARR_TIME_BLOCK_1300-1359', 'ARR_TIME_BLOCK_1400-1459',
        'ARR_TIME_BLOCK_1500-1559', 'ARR_TIME_BLOCK_1600-1659',
        'ARR_TIME_BLOCK_1700-1759', 'ARR_TIME_BLOCK_1800-1859',
        'ARR_TIME_BLOCK_1900-1959', 'ARR_TIME_BLOCK_2000-2059',
        'ARR_TIME_BLOCK_2100-2159', 'ARR_TIME_BLOCK_2200-2259',
        'ARR_TIME_BLOCK_2300-2400']
        
        for col in expected_columns:
            if col not in feature_vector.columns:
                feature_vector[col] = 0.0
                
        feature_vector = feature_vector[expected_columns]
        feature_vector.columns = feature_vector.columns.str.replace(r"[<>\[\],]", "_", regex=True)
        # Convert feature vector to DMatrix
        dtest_inference = xgb.DMatrix(feature_vector)

        # Load the XGBoost model
        xgb_model_inference = xgb.Booster()
        xgb_model_inference.load_model("xgboost_model.json")

        # Perform inference
        y_pred_prob = xgb_model_inference.predict(dtest_inference)
        y_pred_binary = (y_pred_prob > 0.5).astype(int)

        # Return the prediction
        return jsonify({
            "prediction_probability": float(y_pred_prob[0]),
            "prediction_class": int(y_pred_binary[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)









