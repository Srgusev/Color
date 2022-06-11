import os
import sqlite3
from sqlite3 import Connection
import streamlit as st
import pandas as pd


# Подсоединяем БД
URI_SQLITE_DB = "VKRS.db"
#Connecting to sqlite
conn = sqlite3.connect('VKRS.db')
#Creating a cursor object using the cursor() method
cursor = conn.cursor()
#Doping EMPLOYEE table if already exists.
#cursor.execute("DROP TABLE IF EXISTS VKRS")

def init_db(conn: Connection):
    conn.execute(
            """CREATE TABLE IF NOT EXISTS VKRS
            (
                name TEXT,
                size INTEGER,
                type TEXT
            );"""
    )
    conn.commit()

def get_data(conn: Connection):
    df = pd.read_sql("SELECT * FROM VKRS", con=conn)
    return df


def main():
    init_db(conn)
    menu = ["Раскрашивание фотографий"]
    choice = st.sidebar.selectbox("Меню",menu)

    if choice == "Раскрашивание фотографий":
        st.subheader('Пожалуйста загрузите изображение')

        with st.form(key='uploader'):
            uploaded_file = st.file_uploader("Выберите файл...")
            submit_button_upl = st.form_submit_button(label='Раскрасить')
        if (uploaded_file is not None and submit_button_upl):
            st.success('НАЧАЛО')

        elif uploaded_file and submit_button_upl:
            st.header("Черно-белое изображение")
            st.image = (uploaded_file)

        if (uploaded_file and submit_button_upl):
            name_img = uploaded_file.name
            size_img = uploaded_file.__sizeof__()
            type_img = uploaded_file.type
            cursor.execute(f"INSERT INTO VKRS (name,size, type) VALUES ('{name_img}','{size_img} байт','{type_img}')")
            # cursor.execute(f"INSERT INTO vkrb (FIO, DAT, ADRESS, MONEY, FILE) VALUES ('{name}', '{data}', '{addr}','{mon}', '{fname}')")

            conn.commit()


if __name__ == '__main__':
            main()

def get_connection(path: str):
    """Put the connection in cache to reuse if path does not change between Streamlit reruns.
    NB : https://stackoverflow.com/questions/48218065/programmingerror-sqlite-objects-created-in-a-thread-can-only-be-used-in-that-sa
    """
    return sqlite3.connect(path, check_same_thread=False)