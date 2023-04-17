import streamlit as st
from lib.streamlit_helpers import *
from lib.model_loader import *
import time


# Ìnormation app
name_app = 'Avocado app'
version_app = '1.0'
current_time = '2023-04-10 11:16:58'

def loading(times):
    with st.spinner('Đang tải...'):
        time.sleep(times)
    
    st.write('Hoàn tất!')

def create_contact_form():
    # contact form
    with st.sidebar:
        st.write('Send us your feedback!')
        name_input = st.text_input('Your name')
        comment_input = st.text_area('Your comment')
        submitted = st.button('Submit')

        # Nếu người dùng đã gửi đánh giá, thêm đánh giá vào dataframe
        if submitted:
            # Thêm đánh đánh giá người dùng vào file txt
            pass

def create_infomation_app(name_app, version_app, current_time):
    # Infomations app
    st.sidebar.markdown(
        """
        <div style='position: fixed; bottom: 0'>
            <p> """+ name_app +""" - Version: """+ version_app +""" </br>(For predicting avocado prices in the US)</p>
            <p><i>Last Updated: """+ current_time +"""<i/></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_introdution_app():
    st.image('assets/images/avocado.png')
    st.markdown("""
    ## Business Issue
    "Hass" Avocado - a company based in Mexico, specializes in producing various types of avocados sold in the US.
    The company operates in multiple regions of the US with 2 types of avocados, regular and organic,
    packaged according to various standards (Small/Large/XLarge Bags), and have 3 different PLUs (Product Look Up) (4046, 4225, 4770).
    However, they do not have a model to predict avocado prices for business expansion.

    => **Objective**: Build a model to predict the average price of "Hass" avocados in the US => consider expanding production and business

    ## Specific requirements:
    - **Task 1**: USA's Avocado Average Price
    Prediction - Use regression algorithms
    such as Linear Regression, Random Forest,
    XGB Regressor...

    - **Task 2**: Conventional/Organic Avocado
    Average Price Prediction for the future in
    California/NewYork... - Use time series
    algorithms such as ARIMA, Prophet...


    ### USA's Avocado AveragePrice Prediction
    - Using *RandomForestRegressor* to predict avocado prices.
    - The regression model achieved an accuracy of *90%* on the test set.
    - It can be applied in practice to predict the average avocado prices in the US.
    However, the model needs to be carefully used and evaluated regularly to ensure the accuracy of predictions.
    Additionally, when applying the model to new data, it's important to check if the new data is similar to the data used for training.

    ### Average Price Prediction in California & Other
   -  Arima model: California (conventional), Los Angeles (organic), San Francisco (conventional)
    - Facebook Prophet: California (organic)
    - Potential regions: Los Angeles, San Francisco. The results show that avocado prices are trending upward in the next 5 years.
    - In practice, the models have good prediction capabilities for data up to 5 years ago.

    ### Selecting potential regions for organic avocados
    => **Los Angeles** is a potential region due to its large scale (total sales), increasing price trends over the years, and high growth rate.

    => **San Francisco** is a potential region due to its large scale (total sales), increasing price trends over the years, and high growth rate

    ## User guide:
    To use the application, you select the corresponding tabs for the regions you want to predict. Then, choose the options and enter the input parameters to predict avocado prices for a period of 1 to 5 years.

    However, we also want to emphasize that the model achieves only about 90% accuracy on the test set and works well for predicting data up to about 5 years in advance.

    We hope that our application will help you predict avocado prices accurately and efficiently. Thank you for using our application!

    """)
    

def show_about_data():
    st.markdown("""
        - Dữ liệu được lấy trực tiếp từ máy tính tiền của các nhà bán lẻ dựa trên doanh số bán lẻ thực tế của bơ Hass.
        - Dữ liệu đại diện cho dữ liệu lấy từ máy quét bán lẻ hàng tuần cho lượng bán lẻ (National retail volume- units) và giá bơ từ tháng 4/2015 đến tháng 3/2018.
        - Giá Trung bình (Average Price) phản ánh giá trên một đơn vị (mỗi quả bơ), ngay cả khi nhiều đơn vị (bơ) được bán trong bao.
        - Mã tra cứu sản phẩm - Product Lookup codes (PLU’s) trong bảng chỉ dành cho bơ Hass, không dành cho các sản phẩm khác.

        Bao gồm các cột:

        - Date - ngày ghi nhận
        - AveragePrice – giá trung bình của một quả bơ
        - Type - conventional / organic – loại: thông thường/ hữu cơ
        - Region – vùng được bán
        - Total Volume – tổng số bơ đã bán
        - 4046 – tổng số bơ có mã PLU 4046 đã bán
        - 4225 - tổng số bơ có mã PLU 4225 đã bán
        - 4770 - tổng số bơ có mã PLU 4770 đã bán
        - Total Bags – tổng số túi đã bán
        - Small/Large/XLarge Bags – tổng số túi đã bán theo size
        """)
        

