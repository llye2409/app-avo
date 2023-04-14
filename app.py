import streamlit as st
from streamlit_option_menu import option_menu
from lib.streamlit_helpers import *
from lib.model_loader import *
import datetime
from pandas_summary import DataFrameSummary
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot


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
scaler = load_scaler()
df = load_data()
regions = load_region()
# reg_model = load_reg_model()
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
        st.subheader('Dự đoán giá bơ trung bình ở Mỹ với Regression')
        
        # Input
        types = st.selectbox("Type",('conventional', 'organic'),  key="selectbox_types_res")
        min_date = datetime.date(2015, 1, 1)
        max_date = datetime.date(2025, 12, 31)
        date_format = 'YYYY-WW'       
        date = st.date_input("Date", value=datetime.date(2015, 8, 18), min_value=min_date, max_value=max_date, key='date_input_res')
        day = date.day
        month = date.month
        year = date.year
        season = convert_month(month)
        region = st.selectbox("Region",options = regions, key='selectbox_region_res')
        total_volume = st.number_input('Total Volume',value=80043.78, key='number_input_total_volume')

        # Upload file
        st.write('Hoặc tải lên file csv để dự đoán nhiều hơn')
        uploaded_file = st.file_uploader("Choose a file", type=['csv'])
        st.markdown("Để tránh báo lỗi, vui lòng upload file csv theo định dạng mẫu [Download template CSV file](https://drive.google.com/u/0/uc?id=1njn4IW4az9nU51EcYI9k2rgbUO9bHa6E&export=download)")
    
        # Prediction
        if st.button('Start prediction', key='button_res'):

            if uploaded_file is not None:
                # Xử lý dữ liệu upload
                new_data_df = pd.read_csv(uploaded_file)
                st.write('some data')
                st.dataframe(new_data_df.head())
                result_df = processing_new_data(new_data_df, show_download=True)
            
            else:                 
                # Xử lý dữ liệu inputs & predict
                new_data_clean = processing_for_new_ppredict(total_volume, types, year, month, day, season, region)
                result = reg_model.predict(new_data_clean)
                # Show result
                st.code('predicted results: ' + str(result))
                
 
    # Thực hiện dự đoán với model Time series
    with tab2:
        st.subheader('Dự đoán giá bơ trung bình vùng California Time series')
        
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
        st.subheader('Dự đoán giá bơ convetional vùng LosAngeles')

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
        st.subheader('Dự đoán giá bơ organic vùng SanFrancisco')

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
  

elif add_select == 'Regression model':
        
    st.header('Regression model')
    
    # Some data
    st.write('Some data')
    st.write(df.head())

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

    #et_model_metrics.show_metrics()
    st.subheader("Show the evaluation metrics")
    metrics = read_file_txt('data/metrics.txt')
    st.code(metrics)
    st.write('R-square = 0.9109 and low MAE (0.0147), the model is suitable for prediction')
    st.image('assets/images/actual_prediction_regression.png')

    
elif add_select == 'Arima model':

    st.header('Arima Model')

    # Somes data
    df_conventional = data_preparation_conventional()
    df_conventional_arima = df_conventional.copy()
    df_conventional_arima.set_index('Date', inplace=True)
    st.table(df_conventional_arima.head())


    # Plot the prediction and actual values of the ARIMA model
    st.subheader('Plot the prediction and actual values of the ARIMA model')
    st.image('assets/images/arima1.png')
    st.image('assets/images/arima2.png')
    # Show the evaluation metrics
    st.subheader("Show the evaluation metrics")
    metrics = read_file_txt('data/arima_metrics.txt')
    st.code(metrics)


elif add_select == 'Prophet model':
    
    st.header('Facebook Prophec')

    # Some data
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


if st.button('Train model regression', key='trainmodel'):
    
     
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

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    scalers = StandardScaler()
    X_arr = scaler.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)

    # Train model
    from sklearn.ensemble import ExtraTreesRegressor
    et_model = ExtraTreesRegressor()
    et_model.fit(X_train, y_train)

    # Save model
    with open('models/reg_model.pkl', 'wb') as f:
        pickle.dump(et_model, f)

        
