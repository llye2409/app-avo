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
    st.markdown("""
    Chào mừng đến với ứng dụng dự đoán giá bơ cho toàn nước Mỹ! Ứng dụng được phát triển dựa trên các mô hình machine learning như ***XGB regression, ARIMA, Facebook Prophet*** dự đoán giá bơ cho các vùng khác nhau như

- California, 
- San Francisco 
- Los Angeles.

Để sử dụng ứng dụng, bạn chọn các tab tương ứng với các vùng cần dự đoán. Sau đó, lựa chọn các tùy chọn và nhập các thông số đầu vào để dự đoán giá bơ trong khoảng thời gian từ 1 đến 5 năm.

Tuy nhiên, chúng tôi cũng muốn nhấn mạnh rằng mô hình chỉ đạt được độ chính xác khoảng 90% trên tập test, và chỉ hoạt động tốt với dữ liệu dự đoán khoảng 5 năm trở lại.

Chúng tôi hy vọng rằng ứng dụng của chúng tôi sẽ giúp bạn dự đoán giá bơ một cách chính xác và hiệu quả. 
Cảm ơn bạn đã sử dụng ứng dụng của chúng tôi!

    """)
    st.image('assets/images/avocado.png')
    

def show_about_data():
    st.markdown("""
    Dữ liệu được lấy trực tiếp từ máy tính tiền của các nhà bán lẻ dựa trên doanh số bán lẻ thực tế của bơ Hass.

    Dữ liệu đại diện cho dữ liệu lấy từ máy quét bán lẻ hàng tuần cho lượng bán lẻ (National retail volume- units) và giá bơ từ tháng 4/2015 đến tháng 3/2018.
    
    Giá Trung bình (Average Price) trong bảng phản ánh giá trên một đơn vị (mỗi quả bơ), ngay cả khi nhiều đơn vị (bơ) được bán trong bao.
    
    Mã tra cứu sản phẩm - Product Lookup codes (PLU’s) trong bảng chỉ dành cho bơ Hass, không dành cho các sản phẩm khác.
        Some relevant columns in the dataset:

    - Date - The date of the observation
    - AveragePrice - the average price of a single avocado
    - type - conventional or organic
    - year - the year
    - Region - the city or region of the observation
    - Total Volume - Total number of avocados sold
    - 4046 - Total number of avocados with PLU 4046 sold
    - 4225 - Total number of avocados with PLU 4225 sold
    - 4770 - Total number of avocados with PLU 4770 sold
        """)
        

