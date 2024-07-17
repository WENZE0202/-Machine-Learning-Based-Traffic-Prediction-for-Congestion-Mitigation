import logging
from flask import Flask, render_template, request
from colorlog import ColoredFormatter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import joblib
import numpy as np


app = Flask(__name__)

# Configure logging with colored output
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'green',  # Set DEBUG level to green color
        'INFO': 'cyan',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

# Create and configure the logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # Set logger level to DEBUG


@app.route('/')
def index():
    return render_template('index.html')

scaler = MinMaxScaler(feature_range=(0, 1))

def preprocess_data(params):
    logger.debug('Preprocessing data...')

    # Load the final dataset for label encoding and scaling
    df = pd.read_csv('final_dataset.csv')

    # Filter the dataset to include only the columns present in the received parameters
    relevant_columns = ['temp', 'rain_1h', 'snow_bin', 'clouds_all', 'Day_Week', 'Weekend', 'weather_desc', 'holi', 'month', 'hour', 'day_part', 'traffic_volume']
    df_relevant = df[relevant_columns]

    # Label encode categorical columns in the dataset
    le = LabelEncoder()
    for column in df_relevant.columns:
        if df_relevant[column].dtype == 'object':
            df_relevant[column] = le.fit_transform(df_relevant[column])

    # Scale numerical columns using MinMaxScaler

    scaled_data = scaler.fit_transform(df_relevant)

    # Log the received parameters before preprocessing
    logger.debug(f'Received parameters - Temp: {params["temp"]}, Rain 1h: {params["rain_1h"]}, Snow Bin: {params["snow_bin"]}, Clouds: {params["clouds_all"]}, Day of Week: {params["Day_Week"]}, Weekend: {params["Weekend"]}, Weather Desc: {params["weather_desc"]}, Holiday: {params["holi"]}, Month: {params["month"]}, Hour: {params["hour"]}, Day Part: {params["day_part"]}')

    # Convert the received parameters to a DataFrame
    data = pd.DataFrame(params, index=[0])

    # Convert specific columns back to their original data types
    data['temp'] = pd.to_numeric(data['temp'])
    data['rain_1h'] = pd.to_numeric(data['rain_1h'])
    data['snow_bin'] = pd.to_numeric(data['snow_bin'])
    data['clouds_all'] = pd.to_numeric(data['clouds_all'])
    data['Day_Week'] = pd.to_numeric(data['Day_Week'])
    data['Weekend'] = pd.to_numeric(data['Weekend'])
    data['weather_desc'] = pd.to_numeric(data['weather_desc'])
    data['holi'] = pd.to_numeric(data['holi'])
    data['month'] = pd.to_numeric(data['month'])
    data['hour'] = pd.to_numeric(data['hour'])
    data['day_part'] = data['day_part']  # No conversion needed for 'day_part'
    data['traffic_volume'] = 0

    data = data[df_relevant.columns]

    # Apply label encoding and scaling using the same transformers from the final dataset
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.transform(data[column])

    data = scaler.transform(data)


    # Log the scaled data for debugging
    logger.debug(f'Scaled data: {data}')

    # Convert the NumPy array back to a DataFrame
    data_df = pd.DataFrame(data,
                           columns=['temp', 'rain_1h', 'snow_bin', 'clouds_all', 'Day_Week', 'Weekend', 'weather_desc',
                                    'holi', 'month', 'hour', 'day_part','traffic_volume'])

    data_df = data_df.drop(columns=['traffic_volume'])
    print(data_df)
    # Log the parameters after preprocessing
    logger.debug(
        f'Preprocessed parameters - Temp: {data_df.iloc[0]["temp"]}, Rain 1h: {data_df.iloc[0]["rain_1h"]}, Snow Bin: {data_df.iloc[0]["snow_bin"]}, Clouds: {data_df.iloc[0]["clouds_all"]}, Day of Week: {data_df.iloc[0]["Day_Week"]}, Weekend: {data_df.iloc[0]["Weekend"]}, Weather Desc: {data_df.iloc[0]["weather_desc"]}, Holiday: {data_df.iloc[0]["holi"]}, Month: {data_df.iloc[0]["month"]}, Hour: {data_df.iloc[0]["hour"]}, Day Part: {data_df.iloc[0]["day_part"]}')

    logger.debug('Data preprocessing complete.')
    return data_df



@app.route('/predict', methods=['POST'])
def predict():
    logger.debug('Received POST request to /predict')

    # Get parameters from the request form
    temp = request.form['temp']
    rain_1h = request.form['rain_1h']
    snow_bin = request.form['snow_bin']
    clouds_all = request.form['clouds_all']
    Day_Week = request.form['Day_Week']
    Weekend = request.form['Weekend']
    weather_desc = request.form['weather_desc']
    holi = request.form['holi']
    month = request.form['month']
    hour = request.form['hour']
    day_part = request.form['day_part']

    # Log the received parameters
    logger.debug(f'Parameters received - Temp: {temp}, Rain 1h: {rain_1h}, Snow Bin: {snow_bin}, Clouds: {clouds_all}, Day of Week: {Day_Week}, Weekend: {Weekend}, Weather Desc: {weather_desc}, Holiday: {holi}, Month: {month}, Hour: {hour}, Day Part: {day_part}')

    # Prepare parameters for preprocessing
    params = {'temp': temp, 'rain_1h': rain_1h, 'snow_bin': snow_bin, 'clouds_all': clouds_all, 'weather_desc': weather_desc, 'holi': holi, 'Day_Week': Day_Week, 'Weekend': Weekend, 'month': month, 'hour': hour, 'day_part': day_part}

    # Preprocess data and make predictions
    # Load the saved model
    model = joblib.load('best_lgb_model.joblib')
    preprocess = preprocess_data(params)

    # Make predictions using the preprocessed data
    predictions = model.predict(preprocess)
    print(predictions)
    # Create an array with zeros for the non-traffic volume columns
    zeros_array = np.zeros((predictions.shape[0], 11))  # Assuming predictions.shape[0] gives the number of predictions

    # Reshape predictions to have the same number of dimensions as zeros_array
    reshaped_predictions = predictions.reshape(-1, 1)

    # Concatenate the zeros array with the traffic volume predictions
    input_data = np.concatenate((zeros_array, reshaped_predictions), axis=1)

    # Inverse transform the input data to get the original values
    original_values = scaler.inverse_transform(input_data)[:, -1]  # Assuming traffic volume is the last column

    logger.debug(f'Predictions: {original_values}')

    # Convert original_value to integer
    original_value_int = int(original_values)

    # Render the result.html template and pass the original_value_int variable to it
    return render_template('result.html', original_value=original_value_int)


if __name__ == '__main__':
    app.run(debug=True)
