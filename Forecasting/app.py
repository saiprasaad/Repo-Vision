'''
Goal of LSTM microservice:
1. LSTM microservice will accept the GitHub data from Flask microservice and will forecast the data for next 1 year based on past 30 days
2. It will also plot three different graph (i.e.  "Model Loss", "LSTM Generated Data", "All Issues Data") using matplot lib 
3. This graph will be stored as image in Google Cloud Storage.
4. The image URL are then returned back to Flask microservice.
'''
# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from flask_cors import CORS

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.python.keras import Sequential
#For dev use this
# from tensorflow.python.keras.layers import Input, Dense, Dropout, LSTM
#For local use this
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima
from pmdarima.arima.utils import ndiffs

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initlize Google cloud storage client

#For local use this
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'lstm-419919-5456620a5103.json'
os.environ['BUCKET_NAME'] = 'lstm-bucket-ssp24scm43k'
os.environ['BASE_IMAGE_PATH'] = 'https://storage.googleapis.com/lstm-bucket-ssp24scm43k/'
client = storage.Client()

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/test', methods=['GET'])
def test():
    # Return a simple response for GET requests
    return jsonify({"message": "Hello, World!"})

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    data = body["issues"]
    type = body["type"]
    category = body["category"]
    print(category)
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    print(len(test))
    if(len(test) < 30):
        look_back = len(test) - 4
        if(look_back < 0):
            look_back = 1
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


    print(X_train)
    print(X_test)

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    
    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    LOCAL_IMAGE_PATH = "static/images/"

    MODEL_LOSS_IMAGE_NAME = "model_loss_" + category +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + category +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_DATA_IMAGE_NAME = "all_data_" + category + "_"+ repo_name + ".png"
    ALL_DATA_URL = BASE_IMAGE_PATH + ALL_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + category)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
    }

    # Predict issues for test data
    y_pred = model.predict(X_test)
    num_weeks = len(y_pred) // 7
    weekly_predictions = np.sum(y_pred[:num_weeks*7].reshape(num_weeks, 7), axis=0)

    # Find the day of the week with maximum issues created
    max_created_day_index = np.argmax(weekly_predictions) % 7
    max_created_day = day_mapping[max_created_day_index]

    # Plot the LSTM Generated image
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             Y_test, marker='.', label="true")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             y_pred, 'r', label="prediction")
    axs.legend()
    axs.set_title('LSTM Generated Data For ' + category)
    axs.set_xlabel('Time Steps')
    axs.set_ylabel(category)
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('All '+category+' Data')
    axs.set_xlabel('Date')
    axs.set_ylabel(category)
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(MODEL_LOSS_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)
    new_blob = bucket.blob(ALL_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)
    new_blob = bucket.blob(LSTM_GENERATED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    # Construct the response
    json_response = {
        "model_loss_image_url": MODEL_LOSS_URL,
        "lstm_generated_image_url": LSTM_GENERATED_URL,
        "all_issues_data_image": ALL_DATA_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)

@app.route('/api/forecast/max', methods=['POST'])
def forecast_max():
    body = request.get_json()
    data = body["issues"]
    type = body["type"]
    category = body["category"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    print(len(test))
    if(len(test) < 30):
        look_back = len(test) - 4
        if(look_back < 0):
            look_back = 1
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


    print(X_train)
    print(X_test)

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    
    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    LOCAL_IMAGE_PATH = "static/images/"

    MODEL_LOSS_IMAGE_NAME = "model_loss_" + category +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + category +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_DATA_IMAGE_NAME = "all_data_" + category + "_"+ repo_name + ".png"
    ALL_DATA_URL = BASE_IMAGE_PATH + ALL_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + category)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
    }

    # Predict issues for test data
    y_pred = model.predict(X_test)
    num_weeks = len(y_pred) // 7
    weekly_predictions = np.sum(y_pred[:num_weeks*7].reshape(num_weeks, 7), axis=0)
    print(weekly_predictions)
    # Find the day of the week with maximum issues created
    max_created_day_index = np.argmax(weekly_predictions) % 7
    max_created_day = day_mapping[max_created_day_index]
    print("Day of the week with maximum issues created:", max_created_day)


    data = body["issues"]
    type = body["type2"]
    category = body["category"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    print(len(test))
    if(len(test) < 30):
        look_back = len(test) - 4
        if(look_back < 0):
            look_back = 1
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


    print(X_train)
    print(X_test)

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # Model to forecast
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=50, batch_size=80, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    
    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    LOCAL_IMAGE_PATH = "static/images/"

    MODEL_LOSS_IMAGE_NAME = "model_loss_" + category +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + category +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_DATA_IMAGE_NAME = "all_data_" + category + "_"+ repo_name + ".png"
    ALL_DATA_URL = BASE_IMAGE_PATH + ALL_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + category)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
    }

    # Predict issues for test data
    y_pred = model.predict(X_test)

    # Initialize an array to store weekly predictions
    weekly_predictions = [0] * 7

    # Iterate over each prediction and accumulate weekly totals
    for i, pred in enumerate(y_pred):
        day_of_week = i % 7
        weekly_predictions[day_of_week] += pred

    # Find the day of the week with maximum issues created
    max_closed_day_index = np.argmax(weekly_predictions)
    max_closed_day = day_mapping[max_closed_day_index]
    print("Day of the week with maximum issues closed:", max_closed_day)



    y_pred = model.predict(X_test)

    # Initialize an array to store monthly predictions
    monthly_predictions = [0] * len(X_test)  # Assuming there are 12 months in a year

    # Iterate over each prediction and accumulate monthly totals
    for i, pred in enumerate(y_pred):
        monthly_predictions[i] += pred

    # Find the month of the year with maximum issues closed
    max_closed_month_index = np.argmax(monthly_predictions)
    month_mapping = {
    0: 'January',
    1: 'February',
    2: 'March',
    3: 'April',
    4: 'May',
    5: 'June',
    6: 'July',
    7: 'August',
    8: 'September',
    9: 'October',
    10: 'November',
    11: 'December'
    }
    # Map the numerical index to the corresponding month name
    max_closed_month_name = month_mapping[max_closed_month_index % 12]
    print("Month of the year with maximum issues closed:", max_closed_month_name)
    json_response = {
        "day_max_issues_created": max_created_day,
        "day_max_issues_closed": max_closed_day,
        "month_max_issues_closed": max_closed_month_name
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


@app.route('/api/forecast/statsmodel', methods=['POST'])
def forecast_using_statsmodel():
    body = request.get_json()
    data = body["issues"]
    type = body["type"]
    category = body["category"]
    print(category)
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)

    # Declaring the sarimax model for statsmodel prediction
    sarimaxModel = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) 

    # Fitting the model
    results = sarimaxModel.fit()
 
    forecast = results.get_forecast(steps=30)
    statsForecastResults = results
    forecast_values = forecast.predicted_mean
    meanValue = forecast_values = forecast.predicted_mean
    forecast_index = pd.date_range(start=df.index[-1], periods=30 + 1, freq='D')[1:]
    # Storing the index
    forecast_values.index = forecast_index
    indexValue = forecast_values.index

    df['day_of_week'] = df.index.dayofweek
    issues_created_by_day = df.groupby('day_of_week')['y'].sum()
    max_created_day = issues_created_by_day.idxmax()
    print("Day of the week with maximum issues created:", max_created_day)


    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    LOCAL_IMAGE_PATH = "static/images/"

    ALL_DATA_IMAGE_NAME = "stats" + category + "_"+ repo_name + ".png"
    ALL_DATA_URL = BASE_IMAGE_PATH + ALL_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['y'], label='Original Data')
    plt.plot(forecast_values.index, forecast_values, label='Forecast', color='red')
    plt.title("Forecasted values of "+type + "using Statsmodel")
    plt.xlabel('Date')
    plt.ylabel(type)
    plt.legend()
    plt.savefig(LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(ALL_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    json_response = {
        "stats_data_image": ALL_DATA_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


@app.route('/api/forecast/statsmodel/max', methods=['POST'])
def forecast_max_stats():
    body = request.get_json()
    data = body["issues"]
    type = body["type"]
    category = body["category"]
    print(category)
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)

    # Declaring the sarimax model for statsmodel prediction
    sarimaxModel = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) 

    # Fitting the model
    results = sarimaxModel.fit()
 
    forecast = results.get_forecast(steps=30)
    statsForecastResults = results
    forecast_values = forecast.predicted_mean
    meanValue = forecast_values = forecast.predicted_mean
    forecast_index = pd.date_range(start=df.index[-1], periods=30 + 1, freq='D')[1:]
    # Storing the index
    forecast_values.index = forecast_index
    indexValue = forecast_values.index
    

    day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
    }

    df['day_of_week'] = df.index.dayofweek
    issues_created_by_day = df.groupby('day_of_week')['y'].sum()
    max_created_day = issues_created_by_day.idxmax()
    max_created_day = day_mapping[max_created_day % 7]
    print("Day of the week with maximum issues created:", max_created_day)

    body = request.get_json()
    data = body["issues"]
    type = body["type2"]
    category = body["category"]
    print(category)
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)

    # Declaring the sarimax model for statsmodel prediction
    sarimaxModel = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) 

    # Fitting the model
    results = sarimaxModel.fit()
 
    forecast = results.get_forecast(steps=30)
    statsForecastResults = results
    forecast_values = forecast.predicted_mean
    meanValue = forecast_values = forecast.predicted_mean
    forecast_index = pd.date_range(start=df.index[-1], periods=30 + 1, freq='D')[1:]
    # Storing the index
    forecast_values.index = forecast_index
    indexValue = forecast_values.index
    

    day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
    }

    month_mapping = {
    0: 'January',
    1: 'February',
    2: 'March',
    3: 'April',
    4: 'May',
    5: 'June',
    6: 'July',
    7: 'August',
    8: 'September',
    9: 'October',
    10: 'November',
    11: 'December'
    }

    df['day_of_week'] = df.index.dayofweek
    issues_closed_by_day = df.groupby('day_of_week')['y'].sum()
    max_closed_day = issues_closed_by_day.idxmax() + 1
    max_closed_day = day_mapping[max_closed_day % 7]
    print("Day of the week with maximum issues closed:", max_closed_day)


    df['month'] = df.index.month
    issues_closed_by_month = df.groupby('month')['y'].sum() 
    max_closed_month = issues_closed_by_month.idxmax()
    max_closed_month = month_mapping[max_closed_month % 12]
    print("Month with maximum issues closed:", max_closed_month)


    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    LOCAL_IMAGE_PATH = "static/images/"

    ALL_DATA_IMAGE_NAME = "stats" + category + "_"+ repo_name + ".png"
    ALL_DATA_URL = BASE_IMAGE_PATH + ALL_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['y'], label='Original Data')
    plt.plot(forecast_values.index, forecast_values, label='Forecast', color='red')
    plt.title("Forecasted values of "+type + "using Statsmodel")
    plt.xlabel('Date')
    plt.ylabel(type)
    plt.legend()
    plt.savefig(LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(ALL_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    json_response = {
        "day_max_issues_created": max_created_day,
        "day_max_issues_closed": max_closed_day,
        "month_max_issues_closed": max_closed_month
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


@app.route('/api/forecast/prophet', methods=['POST'])
def forecast_using_prophet():
    body = request.get_json()
    data = body["issues"]
    type = body["type"]
    category = body["category"]
    print(category)
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']
    model = Prophet()
    # Fitting the data frame
    model.fit(df)
    result = model.make_future_dataframe(periods=60)  
    prediction_results = model.predict(result)

    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    LOCAL_IMAGE_PATH = "static/images/"

    ALL_DATA_IMAGE_NAME = "prophet" + category + "_"+ repo_name + ".png"
    ALL_DATA_URL = BASE_IMAGE_PATH + ALL_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')
    
    
    # Plotting the prediction results
    fig = model.plot(prediction_results)

    plt.xlabel('Date')
    plt.ylabel(type)
    plt.title("Forecasted values of "+type + "using Prophet")
    fig.savefig(LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(ALL_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(ALL_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    json_response = {
        "prophet_data_image": ALL_DATA_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)

@app.route('/api/forecast/prophet/max', methods=['POST'])
def forecast_using_prophet_max():
    body = request.get_json()
    data = body["issues"]
    type = body["type"]
    category = body["category"]
    print(category)
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']
    model = Prophet()
    # Fitting the data frame
    model.fit(df)
    result = model.make_future_dataframe(periods=60)  
    prediction_results = model.predict(result)

    day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
    }

    month_mapping = {
    0: 'January',
    1: 'February',
    2: 'March',
    3: 'April',
    4: 'May',
    5: 'June',
    6: 'July',
    7: 'August',
    8: 'September',
    9: 'October',
    10: 'November',
    11: 'December'
    }

    df['ds'] = pd.to_datetime(df['ds'])
    created_day_of_week = df['ds'].dt.dayofweek.value_counts().idxmax()
    created_day_of_week = day_mapping[created_day_of_week % 7]
    print("Day of the week with maximum issues created:", created_day_of_week)

    body = request.get_json()
    data = body["issues"]
    type = body["type2"]
    category = body["category"]
    print(category)
    repo_name = body["repo"]
    data_frame = pd.DataFrame(data)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']
    model = Prophet()
    # Fitting the data frame
    model.fit(df)
    result = model.make_future_dataframe(periods=60)  
    prediction_results = model.predict(result)

    day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
    }

    month_mapping = {
    0: 'January',
    1: 'February',
    2: 'March',
    3: 'April',
    4: 'May',
    5: 'June',
    6: 'July',
    7: 'August',
    8: 'September',
    9: 'October',
    10: 'November',
    11: 'December'
    }

    df['ds'] = pd.to_datetime(df['ds'])
    closed_day_of_week = df['ds'].dt.dayofweek.value_counts().idxmax()
    closed_day_of_week = day_mapping[closed_day_of_week % 7]
    print("Day of the week with maximum issues created:", closed_day_of_week)

    closed_month = df['ds'].dt.month.value_counts().idxmax()
    closed_month = month_mapping[closed_month % 12]

    

    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    LOCAL_IMAGE_PATH = "static/images/"

    ALL_DATA_IMAGE_NAME = "prophet" + category + "_"+ repo_name + ".png"
    ALL_DATA_URL = BASE_IMAGE_PATH + ALL_DATA_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')
    
    
    # Plotting the prediction results
    fig = model.plot(prediction_results)

    plt.xlabel('Date')
    plt.ylabel(type)
    plt.title("Forecasted values of "+type + "using Prophet")
    fig.savefig(LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(ALL_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(ALL_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_DATA_IMAGE_NAME)

    json_response = {
        "day_max_issues_created": created_day_of_week,
        "day_max_issues_closed": closed_day_of_week,
        "month_max_issues_closed": closed_month
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)

# Run LSTM app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
