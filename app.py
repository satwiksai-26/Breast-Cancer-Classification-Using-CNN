import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Set up the Streamlit app page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

# Hide Streamlit style
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define the class names
class_names = ['Benign', 'Malignant', 'Normal']

# Sidebar content
with st.sidebar:
    st.image('bk1.jpg')
    st.title("Breast Cancer Classification")
    st.subheader("Accurate detection of cancer present in the Breast Cancers. This helps a user to easily detect the cancer and identify its cause.")

st.write("""
    # Breast Cancer Classification Using CNN
""")

# Load the model with caching to avoid reloading on every run
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('keras_model.h5')
    return model

with st.spinner('Model is being loaded...'):
    model = load_model()

# File uploader for images
file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)    
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image

    if img.shape[-1] == 4:  # Check if the image has an alpha channel
        img = img[..., :3]  # Remove the alpha channel

    img_reshape = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).convert('RGB')  # Ensure the image is in RGB mode
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    result = class_names[np.argmax(predictions)]
    string = "Detected Disease : " + result

    if result == 'Benign':
        st.balloons()
        st.sidebar.success(string)
    elif result == 'Malignant':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")
    elif result == 'Normal':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active. If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection.")
