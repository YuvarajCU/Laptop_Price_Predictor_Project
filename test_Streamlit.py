#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pytest

def test_prediction():
    # Define test inputs
    company = "Acer"
    lap_type = "Ultrabook"
    ram = 8
    weight = 1.2
    touchscreen = 1
    ips = 1
    screen_size = 13.3
    resolution = "1920x1080"
    cpu = "Intel"
    hdd = 0
    ssd = 256
    gpu = "Nvidia"
    os = "Windows 10"

    # Run the prediction
    with st.form("my_form"):
        st.selectbox('Brand', [company])
        st.selectbox("Type", [lap_type])
        st.selectbox("Ram(in GB)", [ram])
        st.number_input("Weight of the Laptop", value=weight)
        st.selectbox("TouchScreen", ['No', 'Yes'], index=touchscreen)
        st.selectbox("IPS", ['No', 'Yes'], index=ips)
        st.number_input('Screen Size', value=screen_size)
        st.selectbox('Screen Resolution',[resolution])
        st.selectbox('CPU',[cpu])
        st.selectbox('HDD(in GB)',[hdd])
        st.selectbox('SSD(in GB)',[ssd])
        st.selectbox('GPU',[gpu])
        st.selectbox('OS',[os])
        submitted = st.form_submit_button("Predict Price")
    assert submitted is True


# In[ ]:




