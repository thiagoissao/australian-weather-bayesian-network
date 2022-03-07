import pandas as pd  # for data manipulation


def get_data_frame(path):
    # Set Pandas options to display more columns
    pd.options.display.max_columns = 50

    # Read in the weather data csv
    dataFrame = pd.read_csv(path, encoding='utf-8')

    # Drop records where target RainTomorrow=NaN
    dataFrame = dataFrame[pd.isnull(dataFrame['RainTomorrow']) == False]

    # For other columns with missing values, fill them in with column mean
    dataFrame = dataFrame.fillna(dataFrame.mean(numeric_only=True))

    # Create bands for variables that we want to use in the model
    dataFrame['Cloud9amCat'] = dataFrame['Cloud9am'].apply(
        lambda x: '1.>4' if x > 4 else '0.<=4')
    dataFrame['Cloud3pmCat'] = dataFrame['Cloud3pm'].apply(
        lambda x: '1.>4' if x > 4 else '0.<=4')
    dataFrame['Humidity9amCat'] = dataFrame['Humidity9am'].apply(
        lambda x: '1.>60' if x > 60 else '0.<=60')
    dataFrame['Humidity3pmCat'] = dataFrame['Humidity3pm'].apply(
        lambda x: '1.>60' if x > 60 else '0.<=60')
    dataFrame['RainfallCat'] = dataFrame['Rainfall'].apply(
        lambda x: '1.>15' if x > 15 else '0.<=15')
    dataFrame['SunshineCat'] = dataFrame['Sunshine'].apply(
        lambda x: '1.>6' if x > 6 else '0.<=6')
    dataFrame['Temp9amCat'] = dataFrame['Temp9am'].apply(
        lambda x: '1.>20' if x > 20 else '0.<=20')
    dataFrame['Temp3pmCat'] = dataFrame['Temp3pm'].apply(
        lambda x: '1.>20' if x > 20 else '0.<=20')
    dataFrame['WindGustSpeedCat'] = dataFrame['WindGustSpeed'].apply(
        lambda x: '0.<=40' if x <= 40 else '1.>40')
    return dataFrame
