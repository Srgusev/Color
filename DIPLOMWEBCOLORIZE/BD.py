import sqlite3
from sqlite3 import Connection
import urllib.request
import base64
import streamlit as st
from io import BytesIO
from matplotlib import pyplot as plt
from colorizers import *
from PIL import Image
import torch
import requests
import colorizers
colorizer_eccv16 = colorizers.eccv16().eval()
colorizer_siggraph17 = colorizers.siggraph17().eval()
import numpy as np
import matplotlib
import pandas as pd

# Подсоединяем БД
#URI_SQLITE_DB = "VKR.db"
#Connecting to sqlite
conn = sqlite3.connect('VKR.db')
#Creating a cursor object using the cursor() method
cursor = conn.cursor()
#Doping EMPLOYEE table if already exists.
#cursor.execute("DROP TABLE IF EXISTS VKR")
#cursor.execute("DROP TABLE IF EXISTS VKROUT")

def init_db(conn: Connection):
    conn.execute(
            """CREATE TABLE IF NOT EXISTS VKR
            (
                name TEXT,
                size REAL,
                type TEXT
            );"""
    )
    conn.execute(
            """CREATE TABLE IF NOT EXISTS VKROUT
            (
                name TEXT,
                size REAL,
                type TEXT
            );"""
    )
    conn.commit()


def get_data(conn: Connection):
    df = pd.read_sql("SELECT * FROM VKR", con=conn)
    return df


def get_dataout(conn: Connection):
    df = pd.read_sql("SELECT * FROM VKROUT", con=conn)
    return df

def display_data(conn: Connection):
    st.header('Просмотр баз данных')
    if st.checkbox("Показать загружаемые изображения"):
        st.dataframe(get_data(conn))


def display_dataout(conn: Connection):
    if st.checkbox("Показать цветные изображения"):
        st.dataframe(get_dataout(conn))


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

def WB_image_download_link(img):
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}" download ="Черно-белое.jpg"><button>Скачать исходное изображение</button></a>'
	return href


def COL_image_download_link(out_img):
	buffered = BytesIO()
	matplotlib.image.imsave(buffered, out_img)
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}" download ="Цвет.jpg"><button>Скачать цветное изображение</button></a>'
	return href

def main():

    init_db(conn)
    menu = ["Раскрашивание фотографий","База данных"]
    choice = st.sidebar.selectbox("Меню",menu)

    if choice == "Раскрашивание фотографий":

        st.subheader('Пожалуйста загрузите изображение')

        with st.form(key='uploader'):
            uploaded_file = st.file_uploader("Выберите файл...")
            url = st.text_input('Вставьте ссылку на изображение')
            submit_button_upl = st.form_submit_button(label='Раскрасить')

        if (uploaded_file and url and submit_button_upl):
            st.error('Ошибка: Загрузите изображение одним способом!')

        elif url and submit_button_upl:
            img = Image.open(requests.get(url, stream=True).raw)
            img = np.array(img)
            img = fix_channels(img)

            with st.spinner(f'Раскраска изображения, подождите...'):
                out_img = process_image(img)
                if (url and submit_button_upl):
                    response = requests.get(url)
                    URLUP = Image.open(BytesIO(response.content))
                    st.write(URLUP)
                    name_img = response.url
                    size_img = urllib.__sizeof__()
                    type_img =  response.headers['Content-Type']
                    cursor.execute(f"INSERT INTO VKR (name,size, type) VALUES ('{name_img}','{size_img} байт','{type_img}')")
                    conn.commit()
                if (url and submit_button_upl):
                    size_img = out_img.__sizeof__()
                    type_img = response.headers['Content-Type']
                    cursor.execute(
                        f"INSERT INTO VKROUT (name,size, type) VALUES ('Цвет','{size_img} байт','{type_img}')")
                    conn.commit()

            col1,col2 = st.columns(2)

            with col1:
                st.header("Черно-белое изображение")
                st.image = (img)
                img = Image.fromarray(img)

                if st.image is not None:
                    col1.image(st.image,use_column_width=True)
                    st.markdown(WB_image_download_link(img),unsafe_allow_html=True)

            with col2:
                st.header("Цветное изображение")
                st.image = (out_img)
        ####3
                #out_img = Image.fromarray((out_img).astype(np.uint8))
                #out_img = Image.fromarray(npimg.astype('uint8'))
        ####3
                if st.image is not None:
                    col2.image(st.image,use_column_width=True)
                    st.markdown(COL_image_download_link(out_img),unsafe_allow_html=True)

        elif uploaded_file and submit_button_upl:
            img = Image.open(uploaded_file)
            img = np.array(img)
            img = fix_channels(img)

            with st.spinner(f'Раскраска изображения, подождите...'):
                out_img = process_image(img)
                if (uploaded_file and submit_button_upl):
                    name_img = uploaded_file.name
                    size_img = uploaded_file.__sizeof__()
                    type_img = uploaded_file.type
                    cursor.execute(f"INSERT INTO VKR (name,size, type) VALUES ('{name_img}','{size_img} байт','{type_img}')")
                    conn.commit()

            col1,col2 = st.columns(2)

            with col1:
                st.header("Черно-белое изображение")
                st.image = (img)
                img = Image.fromarray(img)

                if st.image is not None:
                    col1.image(st.image,use_column_width=True)
                st.markdown(WB_image_download_link(img),unsafe_allow_html=True)

            with col2:
                st.header("Цветное изображение")
                st.image = (out_img)
                size_img = uploaded_file.__sizeof__()
                type_img = uploaded_file.type
                cursor.execute(
                    f"INSERT INTO VKROUT (name,size, type) VALUES ('Цвет','{size_img} байт','{type_img}')")
                conn.commit()
        ####3
                #out_img = Image.fromarray(out_img.astype(np.uint8))
                #np.save('new.jpg', out_img)
                #matplotlib.image.imsave('newfile.jpeg', out_img)
        ####3
                if st.image is not None:
                    col2.image(st.image,use_column_width=True)
                    st.markdown(COL_image_download_link(img),unsafe_allow_html=True)

    if choice == "База данных":
                st.subheader("База Данных")
                display_data(conn)
                display_dataout(conn)

if __name__ == '__main__':
        main()

def get_connection(path: str):
    """Put the connection in cache to reuse if path does not change between Streamlit reruns.
    NB : https://stackoverflow.com/questions/48218065/programmingerror-sqlite-objects-created-in-a-thread-can-only-be-used-in-that-sa
    """
    return sqlite3.connect(path, check_same_thread=False)