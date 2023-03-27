from json import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import locale
from pandas_datareader import data as wb
import yfinance as yf
yf.pdr_override()
from keras.models import load_model
from datetime import datetime
import streamlit as st


start='2009-01-01'
end = datetime.now()
st.title('Stock Trend Prediction')
user_in=st.text_input('Enter Stock Ticker','AAPL')
df = wb.get_data_yahoo(user_in,start=start_date, end=end_date)
st.subheader('Date from 2009-01-01 to yesterday')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,8))
plt.plot(df.Close,label='Original Price')
st.pyplot(fig)

ma50=df.Close.rolling(50).mean()
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

st.subheader('Closing Price vs Time Chart with 50 Days MA')
fig=plt.figure(figsize=(12,8))
plt.plot(df.Close,label='Original Price')
plt.plot(ma50,'r',label='50 Days Moving AVG')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 50,100,200 Days MAs')
fig=plt.figure(figsize=(12,8))
plt.plot(df.Close,label='Original Price')
plt.plot(ma50,'r',label='50 Days Moving AVG')
plt.plot(ma100,'g',label='100 Days Moving AVG')
plt.plot(ma200,'y',label='200 Days Moving AVG')
plt.legend()
st.pyplot(fig)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_arr=scaler.fit_transform(data_training)

model=load_model('stock.h5')
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)

scale=scaler.scale_
scale_fact=1/scale[0]
y_predicted=y_predicted*scale_fact
y_test=y_test*scale_fact

st.subheader('Prediction Vs Original')
fig=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
