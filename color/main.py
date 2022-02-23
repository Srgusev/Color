import tensorflow as tf
import streamlit as st
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models

from PIL import Image
import numpy as np
import time



def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="result.jpg">Скачать результат</a>'
    return href


def main():
    st.set_page_config(layout="centered")
    st.title('Нейронный перенос стиля')
    # st.text(tf.executing_eagerly())
    menu = ["Домашняя страница", "О сервисе"]
    choice = st.sidebar.selectbox("Меню", menu)

    if choice == "Домашняя страница":
        st.subheader("Выберите  изображения:")
        col1, col2 = st.beta_columns(2)

        with col1:
            content_image = st.file_uploader("Выберите изображение для обработки", ['png', 'jpg', 'jpeg'],
                                             help="Перетащите или выберите файл, "
                                                  "ограничение размера - 25 Мб на файл\n\n"
                                                  " • Форматы файлов - PNG, JPG, JPEG.")
            if content_image is not None:
                col1.image(content_image, use_column_width=True)

        with col2:
            style_image = st.file_uploader("Выберите черно-белое изображение", ['png', 'jpg', 'jpeg'],
                                           help="Перетащите или выберите файл, "
                                                "ограничение размера - 25 МБ на файл\n\n"
                                                " • Форматы файлов - PNG, JPG, JPEG.")
            if style_image is not None:
                col2.image(style_image, use_column_width=True)

        col4, col5, col6 = st.beta_columns((1, 3, 1))
        with col5:
            number_of_iterations = st.slider("Выберите число итераций цикла обработки", 10, 500, 20, 1,
                                             format=None, key=None,
                                             help="Число итераций определяет итоговое качество цветного изображения")

        col7, col8, col9 = st.beta_columns((2, 1, 2))
        with col8:
            style_transfer_button = st.button("Начать обработку")

        if style_transfer_button:
            if content_image is not None and style_image is not None:
                best, best_loss = style.run_style_transfer(content_image, style_image, number_of_iterations)
                col10, col11, col12 = st.beta_columns((1, 4, 1))
                with col11:
                    if best.any():
                        col11.image(Image.fromarray(best), use_column_width=True)
                col13, col14, col15 = st.beta_columns((4, 1, 4))
                with col14:
                    if best.any():
                        st.markdown(get_image_download_link(Image.fromarray(best)), unsafe_allow_html=True)
            elif content_image is not None:
                st.error("Ошибка: Изображение с исходным цветом не загружено.")
            elif style_image is not None:
                st.error("Ошибка: Изображение для обработки не загружено.")
            else:
                st.error("Ошибка: Изображение для обработки и изображение с исходным цветом не загружены.")
    else:
        st.subheader("О сервисе")
        st.text("Алгоритм нейронных сетей раскрашивает черно-белое изображение")
        st.text("-----------------------------------------------------------------------------")
        st.text("Сервис разработан студентом группы ИУ5Ц-102Б")
        st.text("Гусевым Сергеем")
        st.text("МГТУ им. Н.Э. Баумана, 2022")


if __name__ == '__main__':
    main()
