import streamlit as st
import numpy as np
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import emoji

st.title('Приложение для сжатия картинок с помощью SVD разложения')
st.header('Пример использования: ')


url = 'https://desu.shikimori.one/uploads/poster/animes/19/main_2x-314bc1a3a049922d9da3de2ea34911ab.webp'
image = io.imread(url)[:, :, 0]
U, sing_values, V = np.linalg.svd(image)

#number_components = st.slider('Выберите число компонент', 1, sing_values.shape[0])

sigma = np.zeros(shape = image.shape)

np.fill_diagonal(sigma, sing_values)


#image_trunc = trunc_U@trunc_sigma@trunc_V
number_components = st.slider('Выберите число компонент', 1, min(sing_values.shape))
left, right = st.columns(2)
with left:
    st.header('Оригинал')
    st.image(image)



trunc_U = U[:, :number_components]
trunc_sigma = sigma[:number_components, :number_components]
trunc_V = V[:number_components, :]

trunc_image = trunc_U@trunc_sigma@trunc_V

my_image = (trunc_image - np.min(trunc_image)) / (np.max(trunc_image) - np.min(trunc_image))
with right:
    st.header(f'{number_components} компонент')
    st.image(my_image, clamp=True)



st.header(f'Теперь попробуйте сами {emoji.emojize(":winking_face:")}')

user_image = st.file_uploader('Загрузите свою картинку')
if user_image:
    image_user = Image.open(user_image)

    image_user = np.array(image_user.convert('L'))

    U, sing_values, V = np.linalg.svd(image_user)

    sigma = np.zeros(shape = image_user.shape)

    np.fill_diagonal(sigma, sing_values)

    number_components_user = st.slider('Выберите число компонент', 1, min(sing_values.shape))
    left_u, right_u = st.columns(2)
    with left_u:
        st.header('Оригинал')
        st.image(image_user)



    trunc_U = U[:, :number_components_user]
    trunc_sigma = sigma[:number_components_user, :number_components_user]
    trunc_V = V[:number_components_user, :]

    trunc_image_user = trunc_U@trunc_sigma@trunc_V
    new_image = (trunc_image_user - np.min(trunc_image_user)) / (np.max(trunc_image_user) - np.min(trunc_image_user))

    with right_u:
        st.header(f'{number_components_user} компонент')
        st.image(new_image, clamp=True)
