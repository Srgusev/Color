import base64
import streamlit as st
from io import BytesIO
from colorizers import *
from PIL import Image
import torch
import requests
import colorizers
colorizer_eccv16 = colorizers.eccv16().eval()
colorizer_siggraph17 = colorizers.siggraph17().eval()
import numpy as np
import matplotlib

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
    # out_img.save(buffered, format="JPEG")

menu = ["Раскрашивание фотографий","О сервисе"]
choice = st.sidebar.selectbox("Меню",menu)

if choice == "Раскрашивание фотографий":

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

    if (uploaded_file and url and submit_button_upl):
        st.error('Ошибка: Загрузите изображение одним способом!')


    elif url and submit_button_upl:
        img = Image.open(requests.get(url, stream=True).raw)
        img = np.array(img)
        img = fix_channels(img)

        with st.spinner(f'Раскраска изображения, подождите...'):
            out_img = process_image(img)
            st.balloons()
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
            #out_img = Image.fromarray((out_img).astype(np.uint8))
            #out_img = Image.fromarray(npimg.astype('uint8'))

            if st.image is not None:
                col2.image(st.image,use_column_width=True)
                st.markdown(COL_image_download_link(out_img),unsafe_allow_html=True)

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
            img = Image.fromarray(img)

            if st.image is not None:
                col1.image(st.image,use_column_width=True)
            st.markdown(WB_image_download_link(img),unsafe_allow_html=True)

        with col2:
            st.header("Цветное изображение")
            st.image = (out_img)
            #out_img = Image.fromarray(out_img.astype(np.uint8))
            #np.save('new.jpg', out_img)
            #matplotlib.image.imsave('newfile.jpeg', out_img)

            if st.image is not None:
                col2.image(st.image,use_column_width=True)
            st.markdown(COL_image_download_link(out_img),unsafe_allow_html=True)

if choice == "О сервисе":
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
                        О сервисе
                        </p>''',unsafe_allow_html=True)
    col1,col2= st.columns(2)
    with col1:
        st.image = Image.open('SunsetWB.jpg')
        if st.image is not None:
            col1.image(st.image,use_column_width=True)

    with col2:
        st.image = Image.open('Sunset.jpeg')
        if st.image is not None:
            col2.image(st.image,use_column_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.image = Image.open('MountWB.jpg')
        if st.image is not None:
            col3.image(st.image,use_column_width=True)

    with col4:
        st.image = Image.open('Mount.jpg')
        if st.image is not None:
            col4.image(st.image,use_column_width=True)

    #img = Image.open('Sunset.jpeg')
    #st.image(img, caption= 'Sunrise by the mountains')
    st.header ('Описание')
    st.text("Сервис по раскрашиванию черно-белых изображений с помощью ансамбля нейронных сетей")
    st.text("Вы можете с помощью данного приложения раскрасить свои черно-белые фотографии")
    st.text("Сервис разработан Гусевым Сергеем")
    st.text("МГТУ им. Н.Э. Баумана, 2022")
