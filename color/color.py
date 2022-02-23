import tensorflow as tf
import streamlit as st
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models

from PIL import Image
import numpy as np
import time


# Подготовка данных
def load_image(image_file):
    max_dim = 512
    img = Image.open(image_file)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)

    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img


def load_and_process_image(image_file):
    img = load_image(image_file)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Входные данные для преобразования изображения должны быть изображением типа "
                               "измерение [1, высота, ширина, канал] или [высота, ширина, канал]")
    if len(x.shape) != 3:
        raise ValueError("Неверные входные данные для преобразования изображения")

    # выполним операцию, обратную предварительной обработке
    # изменение средних значений компонент
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Преобразование BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Слои содержания и стиля, с которых мы будем получать карты признаков
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# Построение модели
def get_model():
    # Загружаем предварительно обученную модель VGG19
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Получим выходные слои, соответствующие слоям стиля и содержимого
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    return models.Model(vgg.input, model_outputs)


# Подсчет потерь содержимого
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


# Подсчет потерь стиля
# Получение матрицы Грама

def gram_matrix(input_tensor):
    # Вычислим число каналов
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


# Итоговые потери
def compute_loss(model, loss_weights, init_image, content_features, gram_style_features):
    content_weight, style_weight = loss_weights  # коэффициенты альфа и бета

    # Пропускаем исходное изображение через нашу нейронную сеть
    model_outputs = model(init_image)
    # Выявляем необходимые признаки
    content_output_features = model_outputs[num_style_layers:]
    style_output_features = model_outputs[:num_style_layers]

    # Величины потерь
    content_score = 0
    style_score = 0

    # Объединяем потери стиля со всех слоёв (формула 5)
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Объединяем потери содержимого со всех слоёв (аналог формулы 5)
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    content_score *= content_weight
    style_score *= style_weight

    # Получаем итоговые потери  # формула 6
    loss = content_score + style_score
    return loss, content_score, style_score


# Полезные функции
# Функция для получения карт признаков
def get_feature_representations(model, content_img, style_img):
    content_image = load_and_process_image(content_img)
    style_image = load_and_process_image(style_img)

    content_outputs = model(content_image)
    style_outputs = model(style_image)

    # Получим карты признаков стиля и содержания с помощью нашей модели
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    return content_features, style_features


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Вычисляем градиенты по входному изображению
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_img,
                       style_img,
                       num_iterations=1000,
                       content_weight=1e3, # альфа
                       style_weight=1e-2): # бета

    # Так как слои модели нам тренировать не нужно, поставим параметр trainable = False
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Получим карты признаков содержания и стиля с указанных выше слоёв
    content_features, style_features = get_feature_representations(model, content_img, style_img)
    # Посчитаем матрицы Грама для карт признаков стиля
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Загрузим исходное изображение (создадим его копию для дальнейшего использования)
    init_image = load_and_process_image(content_img)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Создадим оптимизатор для градиентного спуска
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # Переменные для сохранения наилучшего результата
    best_loss, best_img = float('inf'), None

    loss_weights = (content_weight, style_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'content_features': content_features,
        'gram_style_features': gram_style_features
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    estimated = st.empty()
    latest_iteration = st.empty()
    bar = st.progress(0.0)
    col1, col2, col3 = st.beta_columns((1, 2.5, 1))
    with col2:
        cur_image = st.empty()

    start_time = time.time()

    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, content_score, style_score = all_loss
        opt.apply_gradients([(grads, init_image)])  # применяем вычисленный градиент к пикселям нашего изображения
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)  # ограничиваем пиксель изображения мин и макс знач
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        latest_iteration.text(f'Итерация {i + 1}/{num_iterations}')
        estimated.text('Ожидаемое время выполнения: ∞ с')
        if i > 0:
            cur_speed = i / (time.time() - start_time)
            estimated_time = (num_iterations - i - 1) / cur_speed
            estimated.text('Ожидаемое время выполнения: {:.2f} с'.format(estimated_time))
            bar.progress((i + 1) / num_iterations)
            cur_image.image(Image.fromarray(best_img), use_column_width=True)
        if i + 1 == num_iterations:
            cur_image.empty()

    return best_img, best_loss
