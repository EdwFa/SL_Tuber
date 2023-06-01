import os.path

import plotly.express as px
import cv2
import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras import models, layers
import numpy as np
import PIL.Image


model_path = "models/Tuber_model_75"

st.set_page_config(
    page_title="Analise Lung screen",
    page_icon="🧊",
    initial_sidebar_state="expanded"
)

page_style = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
text-align: center
}}
"""

img_size = (300, 300)

labels = ('Здоров', 'Болен')


def load_img(path_to_img):
    print(path_to_img)
    img = PIL.Image.open(path_to_img)
    img = np.array(img)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.

    img = np.array((img, ))

    print(img.shape)
    return img


@st.cache_resource
def load_model():
    model = models.load_model(os.path.abspath(model_path))
    return model


print_label = lambda x: 'Здоров' if x == '0' else 'Болен'


if __name__ == '__main__':

    st.markdown(page_style, unsafe_allow_html=True)

    st.title("Анализ изображений по туберкулезу")
    model = load_model()

    img = st.file_uploader('Загрузите изображение')
    if img:
        st.image(img, use_column_width=True)
        img = load_img(img)

        predicted_prob = model.predict(img)
        print(predicted_prob[0])
        predicted = [np.argmax(pred) for pred in predicted_prob]
        predicted_prob = [{'label': label, 'value': pred} for label, pred in zip(labels, predicted_prob[0])]
        predicted_prob = pd.DataFrame(predicted_prob, columns=['label', 'value'])
        print(predicted, predicted_prob)

        fig = px.bar(predicted_prob, x='value', y='label', width=300, height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.write(f'Статус = {print_label(predicted[0])}')