import streamlit as st
from tensorflow.keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import numpy as np

# Set the title and description of the app
st.title('Cat vs. Dog Image Classifier')

# Load the trained model (use caching to prevent reloading on every run)
@st.cache_resource
def load_trained_model():
    model = load_model('Image_CNN_model.keras')
    return model

model = load_trained_model()

# Create a file uploader component
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Preprocess the image before prediction
    def preprocess_image(image):
        # Resize the image to match model's expected sizing
        size = (64, 64)  # Your model input size
        image = image.resize(size)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescaling as done during training
        return img_array

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    # Interpret the prediction
    if prediction[0][0] >= 0.5:
        prediction_label = 'Dog'
        probability = prediction[0][0]
    else:
        prediction_label = 'Cat'
        probability = 1 - prediction[0][0]

    confidence_percentage = probability * 100
    
    with col2:
        # Display the prediction result
        st.write(f"### Prediction: {prediction_label}")

        if confidence_percentage > 80:
            st.success(f"High confidence ({confidence_percentage:.2f}%)")
        elif 40 <= confidence_percentage <= 80:
            st.warning(f"Moderate confidence ({confidence_percentage:.2f}%)")
        else:
            st.error(f"Low confidence ({confidence_percentage:.2f}%)")