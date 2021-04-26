import numpy as np
import pandas as pd
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv("crop_prediction_model_one.csv")
crop_dict = {
    "rice": "ধান",
    "maize": "ভুট্টা",
    "chickpea": "মটর কলাই",
    "kidneybeans": "শিম",
    "pigeonpeas": "মটরশুঁটি",
    "mothbeans": "অড়হর",
    "mungbean": "মুগ ডাল",
    "blackgram": "কালো ছোলা ডাল",
    "lentil": "মসুর",
    "pomegranate": "ডালিম",
    "banana": "কলা",
    "mango": "আম",
    "grapes": "আঙ্গুর",
    "watermelon": "তরমুজ",
    "muskmelon": "খরমুজ",
    "apple": "আপেল",
    "orange": "কমলা",
    "papaya": "পেঁপে",
    "coconut": "নারকেল",
    "cotton": "তুলো",
    "jute": "পাট",
    "coffee": "কফি"
}

properties_dict = {
    "নাইট্রোজেন": 'N',
    "ফসফরাস": 'P',
    "পটাশিয়াম": 'K',
    "তাপমাত্রা": 'temperature',
    "আর্দ্রতা": 'humidity',
    "পিএইচ": 'ph',
    "বৃষ্টিপাত": 'rainfall'
}

converts_dict = {
    'N': 'Nitrogen',
    'P': 'Phosphorus',
    'K': 'Potassium',
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'ph': 'pH',
    'rainfall': 'Rainfall',
    'label': 'Crops'
}

graph_dict = {
    "বার গ্রাফ": "Bar Plot",
    "স্ক্যাটার গ্রাফ": 'Scatter Plot',
    "বক্স গ্রাফ":  'Box Plot'
}

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def scatterPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(converts_dict[x], fontsize=22)
    plt.ylabel(converts_dict[y], fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x=x, y=y)
    plt.xlabel(converts_dict[x], fontsize=22)
    plt.ylabel(converts_dict[y], fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def boxPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df)
    sns.despine(offset=10, trim=True)
    plt.xlabel(converts_dict[x], fontsize=22)
    plt.ylabel(converts_dict[y], fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> কোন ফসল চাষ করবেন? </h2>
    </div>
    """
    st.sidebar.title("যেকোনো একটি নির্বাচন করুন")
    select_type = st.sidebar.radio("", ('গ্রাফ', 'ফসল নির্বাচন করুন'))
    st.markdown(html_temp, unsafe_allow_html=True)

    if select_type == "গ্রাফ":

        plot_type = st.selectbox("গ্রাফ এর ধরন বাছাই করুণ", ('বার গ্রাফ', 'স্ক্যাটার গ্রাফ', 'বক্স গ্রাফ'))
        st.subheader("মাটির বিভিন্ন উপাদানের মধ্যে সম্পর্ক")

        x = ""
        y = ""

        if plot_type == 'বার গ্রাফ':
            x = 'label'
            y = st.selectbox("ফসলের মধ্যে তুলনা করতে একটি বৈশিষ্ট্য নির্বাচন করুন",
                ('ফসফরাস', 'নাইট্রোজেন', 'পিএইচ', 'পটাশিয়াম', 'তাপমাত্রা', 'আর্দ্রতা', 'বৃষ্টিপাত'))
        if plot_type == 'স্ক্যাটার গ্রাফ':
            x = st.selectbox("ফসলের মধ্যে তুলনা করতে বৈশিষ্ট্য নির্বাচন করুন",
                ('নাইট্রোজেন', 'ফসফরাস', 'পটাশিয়াম', 'পিএইচ', 'তাপমাত্রা', 'আর্দ্রতা', 'বৃষ্টিপাত'))
            y = st.selectbox("ফসলের মধ্যে তুলনা করতে বৈশিষ্ট্য নির্বাচন করুন",
                ('ফসফরাস', 'নাইট্রোজেন', 'পিএইচ', 'পটাশিয়াম', 'তাপমাত্রা', 'আর্দ্রতা', 'বৃষ্টিপাত'))
        if plot_type == 'বক্স গ্রাফ':
            x = "label"
            y = st.selectbox("ফসলের মধ্যে তুলনা করতে বৈশিষ্ট্য নির্বাচন করুন",
                ('ফসফরাস', 'নাইট্রোজেন', 'পিএইচ', 'পটাশিয়াম', 'তাপমাত্রা', 'আর্দ্রতা', 'বৃষ্টিপাত'))

        if st.button("গ্রাফ চিত্র দেখুন"):
            if plot_type == 'বার গ্রাফ':
                x = 'label'
                y = properties_dict[y]
                barPlotDrawer(x, y)
            if plot_type == 'স্ক্যাটার গ্রাফ':
                x = properties_dict[x]
                y = properties_dict[y]
                scatterPlotDrawer(x, y)
            if plot_type == 'বক্স গ্রাফ':
                x = 'label'
                y = properties_dict[y]
                boxPlotDrawer(x, y)
    
    if select_type == "ফসল নির্বাচন করুন":

        st.header("আপনার জমির জন্য উপযুক্ত ফসলটি জানতে মানগুলি দিন")
        st.subheader("মান দিতে স্লাইডারটি টানুন")
        n = st.slider('নাইট্রোজেন', 0, 140)
        p = st.slider('ফসফরাস', 5, 145)
        k = st.slider('পটাশিয়াম', 5, 205)
        temperature = st.slider('তাপমাত্রা', 8.83, 43.68)
        humidity = st.slider('আর্দ্রতা', 14.26, 99.98)
        ph = st.slider('পিএইচ', 3.50, 9.94)
        rainfall = st.slider('বৃষ্টিপাত', 20.21, 298.56)
        
        if st.button("ফসলটি জানুন"):
            output=predict_crop(n, p, k, temperature, humidity, ph, rainfall)
            res = "“"+ crop_dict[output] + "”"
            st.success('আপনার জমির জন্য উপযুক্ত ফসল {}'.format(res))


if __name__=='__main__':
    main()