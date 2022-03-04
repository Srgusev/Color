import base64
from io import BytesIO
import streamlit as st
from PIL import Image


def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="result.jpg">Скачать результат</a>'
    return href


def main():
    st.markdown('''<p style="font-size: 80px;
                    background: -webkit-linear-gradient(#FFA500, #FF4500);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-family: verdana;
                    font-weight: bold;
                    font-size:42px">
                    Раскрашивание черно-белого изображения
                    </p>''',unsafe_allow_html=True)
    # st.text(tf.executing_eagerly())
    menu = ["Домашняя страница", "О сервисе"]
    choice = st.sidebar.selectbox("Меню", menu)

    if choice == "Домашняя страница":
        st.subheader("Выберите  изображения:")
        col1, col2 = st.columns(2)

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

        col4, col5, col6 = st.columns((1, 3, 1))
        with col5:
            number_of_iterations = st.slider("Выберите число итераций цикла обработки", 10, 500, 20, 1,
                                             format=None, key=None,
                                             help="Число итераций определяет итоговое качество цветного изображения")

        col7, col8, col9 = st.columns((2, 1, 2))
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


text = f'''{small_title('Общая информация')}
<p>
Этот webapp использует AI для окрашивания черно-белых изображений.
Пользователи могут отправить черно-белое изображение
в виде файла или вставить ссылку на URL-адрес.
Архитектура CNN, использованная в этом проекте,вдохновлена работами Ричарда Чжана, Филлипа Изолы, Алексея А. Эфроса </p >
<div>
</div>
'''
st.sidebar.markdown(text, unsafe_allow_html=True)
