import streamlit as st
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from colorizers import *
from PIL import Image
import requests
import streamlit.components.v1 as components
import colorizers
colorizer_eccv16 = colorizers.eccv16().eval()
colorizer_siggraph17 = colorizers.siggraph17().eval()

def process_image(img):
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if(torch.cuda.is_available()):
    	tens_l_rs = tens_l_rs.cuda()
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    return out_img_siggraph17

colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(torch.cuda.is_available()):
	colorizer_siggraph17.cuda()

def fix_channels(img):
    if len(img.shape)==2:
        return np.dstack((img,img,img))
    else:
        return img


st.set_page_config(page_title='Колоризация')

def small_title(x):
    text = f'''<p style="background: -webkit-linear-gradient(#FF4500, #FFA500);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-family: verdana;
                        font-weight: bold;
                        font-size:24px">
                        {x}
                        </p>'''
    return text

def html_links(text, link):
    return f'''<a href="{link}" target="_blank">{text}</a>'''

style = '''font_size: 14px;
           color: #aaa;'''

st.sidebar.title("Информация")
img_width = '60px'

text = f'''{small_title('Приложение')}
<p style="{style}">Этот webapp использует AI для окрашивания черно-белых изображений.
Пользователи могут отправить черно-белое изображение в виде файла или вставить ссылку на URL-адрес (убедитесь, что URL-адрес заканчивается расширением файла изображения). </p >
{small_title ('Ссылки')}
<p style="{style}">Архитектура CNN, использованная в этом проекте, вдохновлена работами Ричарда Чжана, Филлипа Изолы, Алексея А. Эфроса.
Подробнее о проекте можете прочитать по ссылке: <a href="http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf"> ЗДЕСЬ </a>
</p>
<div>
</div>
'''
st.sidebar.markdown(text, unsafe_allow_html=True)

st.markdown('''<p style="font-size: 80px;
                background: -webkit-linear-gradient(#FFA500, #FF4500);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: verdana;
                font-weight: bold;
                font-size:32px">
                Раскрашивание фотографий.
                </p>''', unsafe_allow_html=True)

st.subheader('Пожалуйста загрузите изображение')

with st.form(key='uploader'):
    uploaded_file = st.file_uploader("Выберите файл...")
    url = st.text_input('Вставьте ссылку на изображение')
    submit_button_upl = st.form_submit_button(label='Раскрасить')

if (uploaded_file is None and url is None and submit_button_upl):
    st.subheader('Что-то пошло не так, попробуйте ещё раз!')

elif (uploaded_file and url and submit_button_upl):
    st.subheader('Что-то пошло не так, попробуйте ещё раз!')

elif url and submit_button_upl:
    img = Image.open(requests.get(url, stream=True).raw)
    img = np.array(img)
    img = fix_channels(img)

    with st.spinner(f'Раскраска изображения, подождите...'):
        out_img = process_image(img)
    col1,col2 = st.columns(2)

    with col1:
        st.header("Черно-белое изображение")
        st.image = (img)

        if st.image is not None:
            col1.image(st.image,use_column_width=True)

    with col2:
        st.header("Цветное изображение")
        st.image = (out_img)

        if st.image is not None:
            col2.image(st.image,use_column_width=True)




elif uploaded_file and submit_button_upl:
    img = Image.open(uploaded_file)
    img = np.array(img)
    img = fix_channels(img)
    with st.spinner(f'Раскраска изображения, подождите...'):
        out_img = process_image(img)
    col1,col2 = st.columns(2)

    with col1:
        st.header("Черно-белое изображение")
        st.image = (img)

        if st.image is not None:
            col1.image(st.image,use_column_width=True)

    with col2:
        st.header("Цветное изображение")
        st.image = (out_img)

        if st.image is not None:
            col2.image(st.image,use_column_width=True)