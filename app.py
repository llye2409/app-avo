import streamlit as st
from streamlit_option_menu import option_menu
from lib.streamlit_helpers import *
from lib.model_loader import *
import datetime
from pandas_summary import DataFrameSummary
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# Page setting
st.set_page_config(page_title="My App", page_icon=":🥑:", layout="centered")

# Show menu
with st.sidebar:  
    add_select = option_menu('Wellcome!',
                          ['Start prediction', 'Introduction', 'Data Exploration', 'Regression model', 'Arima model', 'Prophet model'],
                          icons=['book', 'key', 'moon', 'calculator', 'pen', 'sun'],
                          default_index=0)

# Contact form
create_contact_form()
# Infomations app
create_infomation_app(name_app, version_app, current_time)
# ------- lOAD MODEL----------
df = load_data()
scaler = load_scaler()
regions = load_region()
# reg_model = load_reg_model()
rf_model_avocado_totalVolume = load_rf_model_avocado_totalVolume()
stepwise_model_conventional_cali = load_stepwise_model_conventional_cali_model()
model_prohet_ogranic_cali = load_model_prohet_ogranic_cali_cali_model()
stepwise_model_ogranic_ARIMA_sa = load_stepwise_model_ogranic_ARIMA_sa()
stepwise_model_conventional_Arima_la = load_stepwise_model_conventional_Arima_la()
df_timeseries = load_data_timeseries()

# =========================GUI=========================
if add_select == 'Start prediction':
    st.title('Predicting the price of avocado 🥑')
    tab1, tab2, tab3, tab4 = st.tabs(["🇺🇸 Total US", "🇨🇦 California", "🇱🇦 LosAngeles", "🇸🇦 SanFrancisco"])

    # Thực hiện dự đoán với model regression
    with tab1:
        st.subheader('Predicting the average avocado price in the US with Regression')
        
        # # Input
        # types = st.selectbox("Type",('conventional', 'organic'),  key="selectbox_types_res")
        
        # min_date = datetime.date(2017, 4, 1)
        # max_date = datetime.date(2023, 4, 2)
        # date_format = 'YYYY-WW'       
        # date = st.date_input("Date", value=datetime.date(2018, 4, 1), min_value=min_date, max_value=max_date, key='date_input_res')
        # day = date.day
        # month = date.month
        # year = date.year
        
        # season = convert_month(month)
        
        ## region = st.selectbox("Region",options = regions, key='selectbox_region_res')
        
        # df_for_predict_totalVolume = processing_for_ppredict_totalVolume(types, year, month, day, season, region)
        # total_volume_suggest = rf_model_avocado_totalVolume.predict(df_for_predict_totalVolume)
        # total_volume = st.slider('Select a value', 100, 2300000, int(total_volume_suggest), step=None, key='slider_input_total_volume')

        # # Upload file
        # st.write('Or upload a CSV file to predict more.')
        # uploaded_file = st.file_uploader("Choose a file", type=['csv'])
        # st.markdown("To avoid errors, please upload a CSV file in the specified format [Download template CSV file](https://drive.google.com/u/0/uc?id=1Kv5yM2s6QLu5sWdE4Qv784cE_pWja533&export=download)")
    
        # #Prediction
        # if st.button('Start prediction', key='button_res'):

        #     if uploaded_file is not None:
        #         # Xử lý dữ liệu upload
        #         new_data_df = pd.read_csv(uploaded_file)
        #         st.write('Display some of your data')
        #         st.table(new_data_df.head())
        #         result_df = processing_new_data(new_data_df)

        #         # Show result
        #         st.write('Results:')
        #         st.dataframe(result_df.head())

        #         # Show download file csv
        #         data = result_df.to_csv().encode('utf-8')
        #         st.download_button(label="Download data as CSV", data=data, file_name='data.csv', mime='text/csv')
                
        #     else:                 
        #         # Xử lý dữ liệu inputs & predict
        #         new_data_clean = processing_for_new_ppredict(total_volume, types, year, month, day, season, region)
        #         result = reg_model.predict(new_data_clean)
        #         # Show result
        #         st.code('predicted results: ' + str(result))
                            
 
    # Thực hiện dự đoán với model Time series
    with tab2:
        st.subheader('Predicting the average avocado price in California using Time Series')
        
        # Input
        types = st.radio("Type",('conventional', 'organic'), key="radio_types_timeseries_ca")
        next_years = st.slider("Select a number of years to predict", 1, 5, 3, key='slider_next_years_ca')
        next_times = next_years * 12  # Convert years to months

        # Predict & show results
        if st.button('Start prediction', key='button_timeseries_ca'):

            # Áp dụng Aimara
            if types == 'conventional':
                # Show result           
                st.write('Show results:')
                fig, ax = plot_predicted_prices_Arima(stepwise_model_conventional_cali, next_times, show_table=True, show_download=True)
                st.pyplot(fig)
            
            # Áp dụng Facebook Prophec
            else:
                # Show result           
                st.write('Show results:')
                fig, ax = plot_predicted_prices_Prophec(next_times, show_table=True, show_download=True)
                st.pyplot(fig)

    
    with tab3:
        st.subheader('Predicting the average price of conventional avocados in Los Angeles')

        # Input
        next_years = st.slider("Select a number of years to predict", 1, 5, 3, key='slider_next_years_la')
        next_times = next_years * 12  # Convert years to months
        # Predict & show results
        if st.button('Start prediction', key='button_timeseries_la'):
            # Show result           
                st.write('Show results:')
                fig, ax = plot_predicted_prices_Arima(stepwise_model_conventional_Arima_la, next_times, show_table=True)
                st.pyplot(fig)

    with tab4:
        st.subheader('Predict organic avocado prices in San Francisco')

        # Input
        next_years = st.slider("Select a number of years to predict", 1, 5, 3, key='slider_next_years_sa')
        next_times = next_years * 12  # Convert years to months
        # Predict & show results
        if st.button('Start prediction', key='button_timeseries_sa'):
            # Show result           
                st.write('Show results:')
                fig, ax = plot_predicted_prices_Arima(stepwise_model_ogranic_ARIMA_sa, next_times, show_table=True)
                st.pyplot(fig)

       
elif add_select == 'Introduction':
    # Introdution
    st.title('Introdution')
    create_introdution_app()
elif add_select == 'Data Exploration':
    
    st.header('Data Exploration')

    # Show data
    st.write('Some data')
    st.dataframe(df.sample(5))
    st.subheader('About dataset')
    show_about_data()
    
    st.subheader('Summary')
    # pandas-summary
    report = DataFrameSummary(df)
    st.table(report.summary())

    # Display a histogram of avocado prices
    st.subheader("Distribution of avocado prices")
    fig, ax = plt.subplots()
    sns.histplot(df["AveragePrice"])
    st.pyplot(fig)

    # Display a scatterplot of avocado prices vs. total volume, with different colors for organic and conventional
    st.subheader("Avocado prices vs. total volume, by type")
    fig, ax = plt.subplots()
    sns.scatterplot(x="AveragePrice", y="TotalVolume", hue='type', data=df)
    st.pyplot(fig)

    # Display a boxplot of avocado prices by region
    st.subheader("Avocado prices by region")
    fig, ax = plt.subplots()
    sns.boxplot(x="region", y="AveragePrice", data=df)
    plt.xticks(rotation=90)
    st.pyplot(fig)
  

elif add_select == 'Regression model':
        
    st.header('Regression model')
    
    # Some data
    st.write('Some data')
    st.write(df.head())

    # Show metris models
    st.subheader("Show the evaluation metrics")
    metrics = pd.read_csv('data/metris_regressioin.csv')
    st.table(metrics)
    st.write('RandomForestRegressor is best model with R-square = 0.89 and low MAE (0.02), the model is suitable for prediction')
    
    st.subheader("Visualizing results")
    st.image('assets/images/actual_prediction_regression.png')

    col1, col2 = st.columns(2)
    with col1:
        st.image('assets/images/plot_result_regression_1.png')
        st.image('assets/images/plot_result_regression_2.png')

    with col2:
        st.image('assets/images/plot_result_regression_3.png')
        st.image('assets/images/plot_result_regression_4.png')   

    
elif add_select == 'Arima model':

    st.header('Arima Model')

    # Somes data
    st.write ('Some data')
    df_conventional = data_preparation_conventional()
    df_conventional_arima = df_conventional.copy()
    df_conventional_arima.set_index('Date', inplace=True)
    st.table(df_conventional_arima.head())


    # ARIMA for Avocado Convetional in California
    st.markdown('## 1. ARIMA for Avocado Convetional in California')
    st.markdown('### Plot the prediction and actual values of the ARIMA model')
    st.image('assets/images/arima1.png')
    st.image('assets/images/arima2.png')
    # Show the evaluation metrics
    st.subheader("Show the evaluation metrics")
    metrics = read_file_txt('data/arima_metrics.txt')
    st.code(metrics)

    # ARIMA for Avocado Convetional in Losangeles
    st.markdown('## 1. ARIMA for Avocado organic in Losangeles')
    st.markdown('### Plot the prediction and actual values of the ARIMA model')
    st.image('assets/images/arima_Losangeles1.png')
    st.image('assets/images/arima_Losangeles2.png')
    # Show the evaluation metrics
    st.subheader("Show the evaluation metrics")
    metrics = read_file_txt('data/arima_Losangeles.txt')
    st.code(metrics)

    # ARIMA for Avocado Convetional in Sanfrancisco
    st.markdown('## 1. ARIMA for Avocado Convetional in Sanfrancisco')
    st.markdown('### Plot the prediction and actual values of the ARIMA model')
    st.image('assets/images/arima_Sanfrancisco1.png')
    st.image('assets/images/arima_Sanfrancisco2.png')
    # Show the evaluation metrics
    st.subheader("Show the evaluation metrics")
    metrics = read_file_txt('data/arima_Sanfrancisco.txt')
    st.code(metrics)


elif add_select == 'Prophet model':
    
    st.header('Facebook Prophet')

    # Some data
    st.write('Some data')
    df_organic = data_preparation_organic()
    # create data for Prophet model
    df_organic_prophet = df_organic.copy()
    df_organic_prophet.reset_index(drop=True, inplace=True)
    st.dataframe(df_organic_prophet.head())

    # Plot the prediction and actual values of the ARIMA model
    st.subheader('Plot the prediction and actual values of the Facebook Prophec model')
    st.image('assets/images/prophec1.png')
    st.image('assets/images/prophec2.png')
    st.image('assets/images/prophec3.png')
    st.image('assets/images/prophec4.png')
    
    # Show the evaluation metrics
    st.subheader("Show the evaluation metrics")
    metrics = read_file_txt('data/prophec_metrics.txt')
    st.code(metrics)


# if st.button('Train model regression', key='trainmodel'):
    
#     # Train model
#     X_col = ['TotalVolume', 'type', 'year', 'month', 'day', 'Season', 'region']
#     X = df[X_col]
#     y = df['AveragePrice']

#     # Label Encoder for 'type'
#     X['type'] = X['type'].replace({'conventional': 0, 'organic': 1})

#     # categorical data type conversion
#     lst_categories = ['type', 'region']
#     for col in lst_categories:
#         X[col] = pd.Categorical(X[col])

#     # convert categorical attribute to numeric type: get_dummies()
#     X = dummies('region',X)

#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import train_test_split
#     scalers = StandardScaler()
#     X_arr = scaler.fit_transform(X)
#     X = pd.DataFrame(X_arr, columns=X.columns)

#     # Train test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)

#     # Train model
#     from sklearn.ensemble import ExtraTreesRegressor
#     et_model = ExtraTreesRegressor()
#     et_model.fit(X_train, y_train)

#     # Save model
#     with open('models/reg_model.pkl', 'wb') as f:
#         pickle.dump(et_model, f)


        





if st.button('pre-train model'):

    X_col = ['type', 'year', 'month', 'day', 'Season', 'region']
    X = df[X_col]
    y = df['TotalVolume']

    # Label Encoder for 'type'
    X['type'] = X['type'].replace({'conventional': 0, 'organic': 1})

    # categorical data type conversion
    lst_categories = ['type', 'region']
    for col in lst_categories:
        X[col] = pd.Categorical(X[col])

    # convert categorical attribute to numeric type: get_dummies()
    X = dummies('region',X)

    # Define the model RandomForestRegressor
    rf_model_totalVolumne = RandomForestRegressor()
    rf_model_totalVolumne.fit(X, y)

    # Save model
    with open('models/rf_model_avocado_totalVolume.pkl', 'wb') as f:
        pickle.dump(rf_model_totalVolumne, f)


    # Train model
    X_col = ['TotalVolume', 'type', 'year', 'month', 'day', 'Season', 'region']
    X = df[X_col]
    y = df['AveragePrice']

    # Label Encoder for 'type'
    X['type'] = X['type'].replace({'conventional': 0, 'organic': 1})

    # categorical data type conversion
    lst_categories = ['type', 'region']
    for col in lst_categories:
        X[col] = pd.Categorical(X[col])

    # convert categorical attribute to numeric type: get_dummies()
    X = dummies('region',X)

    scalers = StandardScaler()
    X_arr = scaler.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns)

    # Train model
    from sklearn.ensemble import RandomForestRegressor
    # Define the model RandomForestRegressor
    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)

    # Save model
    with open('models/reg_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
