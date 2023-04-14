import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet.plot import add_changepoints_to_plot
from lib.streamlit_helpers import *
import streamlit as st

# Khởi tạo biến toàn cục để lưu model và dữ liệu
reg_model = None
stepwise_model_conventional_cali = None
model_prohet_ogranic_cali = None
stepwise_model_ogranic_ARIMA_sa = None
stepwise_model_conventional_Arima_la = None
df = None
df_timeseries = None
df_conventional = None
df_organic = None
regions = None
scaler = None


def load_reg_model():
    global reg_model
    
    # Nếu model đã được load trước đó thì không cần load lại
    if reg_model is not None:
        return reg_model
    
    # Nếu model chưa được load thì load model từ file pkl và lưu vào biến toàn cục
    pkl_filename = 'models/reg_model.pkl'
    with open(pkl_filename, 'rb') as file:  
        reg_model = pickle.load(file)
    
    return reg_model

def load_stepwise_model_conventional_cali_model():
    global stepwise_model_conventional_cali
    
    # Nếu model đã được load trước đó thì không cần load lại
    if stepwise_model_conventional_cali is not None:
        return stepwise_model_conventional_cali
    
    # Nếu model chưa được load thì load model từ file pkl và lưu vào biến toàn cục
    pkl_filename = 'models/stepwise_model_conventional_cali.pkl'
    with open(pkl_filename, 'rb') as file:  
        stepwise_model_conventional_cali = pickle.load(file)
    
    return stepwise_model_conventional_cali

def load_model_prohet_ogranic_cali_cali_model():
    global model_prohet_ogranic_cali
    
    # Nếu model đã được load trước đó thì không cần load lại
    if model_prohet_ogranic_cali is not None:
        return model_prohet_ogranic_cali
    
    # Nếu model chưa được load thì load model từ file pkl và lưu vào biến toàn cục
    pkl_filename = 'models/model_prohet_ogranic_cali.pkl'
    with open(pkl_filename, 'rb') as file:  
        model_prohet_ogranic_cali = pickle.load(file)
    
    return model_prohet_ogranic_cali

def load_stepwise_model_ogranic_ARIMA_sa():
    global stepwise_model_ogranic_ARIMA_sa
    
    # Nếu model đã được load trước đó thì không cần load lại
    if stepwise_model_ogranic_ARIMA_sa is not None:
        return stepwise_model_ogranic_ARIMA_sa
    
    # Nếu model chưa được load thì load model từ file pkl và lưu vào biến toàn cục
    pkl_filename = 'models/stepwise_model_ogranic_ARIMA_sa.pkl'
    with open(pkl_filename, 'rb') as file:  
        stepwise_model_ogranic_ARIMA_sa = pickle.load(file)
    
    return stepwise_model_ogranic_ARIMA_sa

def load_stepwise_model_conventional_Arima_la():
    global stepwise_model_conventional_Arima_la
    
    # Nếu model đã được load trước đó thì không cần load lại
    if stepwise_model_conventional_Arima_la is not None:
        return stepwise_model_conventional_Arima_la
    
    # Nếu model chưa được load thì load model từ file pkl và lưu vào biến toàn cục
    pkl_filename = 'models/stepwise_model_conventional_Arima_la.pkl'
    with open(pkl_filename, 'rb') as file:  
        stepwise_model_conventional_Arima_la = pickle.load(file)
    
    return stepwise_model_conventional_Arima_la

st.cache
def load_data():
    global df
    
    # Nếu dữ liệu đã được đọc trước đó thì không cần đọc lại
    if df is not None:
        return df
    
    # Nếu dữ liệu chưa được đọc thì đọc dữ liệu từ file CSV và lưu vào biến toàn cục
    csv_filename = 'data/avocado_new.csv'
    df = pd.read_csv(csv_filename)
    
    return df

st.cache
def load_data_timeseries():
    global df_timeseries

    # Nếu dữ liệu đã được đọc trước đó thì không cần đọc lại
    if df_timeseries is not None:
        return df_timeseries
    
    # Nếu dữ liệu chưa được đọc thì đọc dữ liệu từ file CSV và lưu vào biến toàn cục
    csv_filename = 'data/avocado.csv'
    df_timeseries = pd.read_csv(csv_filename)
   
    return df_timeseries

st.cache
def data_preparation_organic():
    global df_organic, df_timeseries

    # Nếu dữ liệu đã được đọc trước đó thì không cần đọc lại
    if df_organic is not None:
        return df_organic
    
    # create data organic
    df_organic = df_timeseries[(df_timeseries.type == 'organic') & (df_timeseries.region == 'California')][['Date', 'AveragePrice']]
    # Group theo month
    agg = {'AveragePrice': 'mean'}
    df_organic = df_organic.groupby(df_organic['Date']).aggregate(agg).reset_index()
    
    return df_organic

st.cache
def data_preparation_conventional():
    global df_conventional, df_timeseries

    # Nếu dữ liệu đã được đọc trước đó thì không cần đọc lại
    if df_conventional is not None:
        return df_conventional
    
    # create data organic
    df_conventional = df_timeseries[(df_timeseries.type == 'conventional') & (df_timeseries.region == 'California')][['Date', 'AveragePrice']]
    # Group theo month
    agg = {'AveragePrice': 'mean'}
    df_conventional = df_conventional.groupby(df_conventional['Date']).aggregate(agg).reset_index()
    
    return df_conventional


def load_region():
    global regions

    if regions is not None:
        return regions
    
    # Nếu dữ liệu chưa được đọc
    with open('data/regions.txt', 'r') as f:
        regions = f.read().splitlines()

    return regions


def load_scaler():
    global scaler
    
    # Nếu scaler chưa tải
    if scaler is not None:
        return scaler
    
    # Nếu scaler chưa được load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return scaler

def processing_for_new_ppredict(total_volume, types, year, month, day, season, region):
    global scaler, regions
    
    # label coder
    types_avocado = 0 if types == 'conventional' else 1
    # tạo dataframe
    new_data = pd.DataFrame([{'TotalVolume': total_volume,
                        'type': types_avocado,
                        'year': year,
                        'month': month,
                        'day': day,
                        'Season': season}])

    # Create region columns
    for i in regions:
        new_data[i] = 1 if i == region else 0

    # Scaler
    new_data_arr = scaler.transform(new_data)
    new_data_clean = pd.DataFrame(new_data_arr, columns=new_data.columns)
    
    return new_data_clean


def plot_predicted_prices_Arima(model, next_times, show_table=False, show_download=False):
    n_periods = 39

    # Predict next `next_times` months
    future_price_times = model.predict(n_periods=n_periods + next_times)
    future_price_times_df = pd.DataFrame(future_price_times, columns=['Prediction'])

    # Add a "Date" column to the DataFrame
    last_month = pd.to_datetime('2017-06-01')  # Last month in the training data
    future_dates = pd.date_range(start=last_month, periods=len(future_price_times_df), freq='MS')
    future_price_times_df['Date'] = future_dates

    # Compute coefficients of linear regression
    x = np.arange(len(future_price_times_df))
    y = future_price_times_df['Prediction']
    coef = np.polyfit(x, y, 1)

    # Create linear regression object
    trendline = np.poly1d(coef)

    # Create a list of years
    years = pd.date_range(start='2015-01-01', periods=len(future_price_times_df), freq='MS').year

    # Plot the predicted prices and trendline
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, label='Predicted')
    ax.plot(x, trendline(x), 'r--', label='Trendline')
    ax.axvline(x=n_periods, color='r', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Predicted Average Price of Conventional Avocado in California')
    ax.set_xticks(x[::12])
    ax.set_xticklabels(years[::12])
    ax.legend()

    # Show table if requested
    if show_table:
        st.write(future_price_times_df.set_index('Date'))
    
    if show_download:
        data = future_price_times_df.to_csv().encode('utf-8')
        st.download_button(label="Download data as CSV", data=data, file_name='data.csv', mime='text/csv')

    plt.close()  # Close the plot to avoid duplicate plots in Streamlit

    return fig, ax


def plot_predicted_prices_Prophec(next_times, show_table=False, show_download=False):
    global model_prohet_ogranic_cali

    # Create a dataframe with the next 60 months
    future_next_times = model_prohet_ogranic_cali.make_future_dataframe(periods=10 + int(next_times), freq='MS')

    # Make a forecast for the next 60 months
    forecast_next_times = model_prohet_ogranic_cali.predict(future_next_times)

    # make a dataframe forcecast for user
    forecast_next_times_df = forecast_next_times[['ds', 'trend', 'yhat']]

    # Visualize the forecast for the next 60 months
    fig = model_prohet_ogranic_cali.plot(forecast_next_times, xlabel='Date', ylabel='Price')
    ax = add_changepoints_to_plot(fig.gca(), model_prohet_ogranic_cali, forecast_next_times)

    # Show table if requested
    if show_table:
        st.write(forecast_next_times_df)

    if show_download:
        data = forecast_next_times_df.to_csv().encode('utf-8')
        st.download_button(label="Download data as CSV", data=data, file_name='data.csv', mime='text/csv')

    plt.close()  # Close the plot to avoid duplicate plots in Streamlit

    return fig, ax

def processing_new_data(new_data_df, show_download=False):
    global scaler, reg_model
    
    X_col = ['TotalVolume', 'type', 'year', 'month', 'day', 'Season', 'region']
    X_new_data = new_data_df[X_col]

    # Label Encoder for 'type'
    X_new_data['type'] = X_new_data['type'].replace({'conventional': 0, 'organic': 1})

    # Encoder regions
    for idx, row in X_new_data.iterrows(): # duyệt qua từng dòng trong DataFrame
        region = row['region'] # lấy giá trị region của dòng hiện tại
        for r in regions:
            if r == region:
                X_new_data.loc[idx, r] = 1 # cập nhật giá trị 1 cho cột tương ứng với region
            else:
                X_new_data.loc[idx, r] = 0 # cập nhật giá trị 0 cho các cột khác
                
    # Drop region
    X_new_data.drop(columns='region', inplace=True)

    # Scaler
    X_arr = scaler.transform(X_new_data)
    X_new_data = pd.DataFrame(X_arr, columns=X_new_data.columns)

    # predict
    arr_predicted = reg_model.predict(X_new_data)
    # Result
    result_df = new_data_df[['TotalVolume', 'region']]
    result_df['Prediction'] = arr_predicted
    
    # Show result
    st.write('Results:')
    st.dataframe(result_df.head())

    # Show download file csv
    if show_download:
        data = result_df.to_csv().encode('utf-8')
        st.download_button(label="Download data as CSV", data=data, file_name='data.csv', mime='text/csv')


def read_file_txt(file_name):
    with open(file_name, 'r') as f:
        file_contents = f.read()
    
    return file_contents


def convert_month(month):
    if month == 3 or month == 4 or month == 5:
        return 0
    elif month == 6 or month == 7 or month == 8:
        return 1
    elif month == 9 or month == 10 or month == 11:
        return 2
    else:
        return 3
    

def dummies(x,df):
    temp = pd.get_dummies(df[x])
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
    
