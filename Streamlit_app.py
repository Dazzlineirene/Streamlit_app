import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Download the VGG16 model pre-trained on the ImageNet dataset
# 'include_top=True' means we want the final classification layer too
model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)

# 2. Save it to your project folder
model.save('my_model.h5')

print("Success! 'my_model.h5' has been created in your project folder.")
# Set page title
st.set_page_config(page_title="Image Classifier", layout="centered")


# --- 1. Load the Model ---
@st.cache_resource
def load_my_model():
    # Replace with your actual model filename
    model = tf.keras.models.load_model('my_model.h5')
    return model


model = load_my_model()

# --- 2. App Interface ---
st.title("🚀 Image Classification App")
st.write("Upload an image to see the model's prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    st.write("### Classifying...")


    # --- 3. Preprocessing Logic ---
    def preprocess_image(img):
        # Resize image to match model input (e.g., 224x224 for VGG16)
        size = (224, 224)
        img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.asarray(img)
        img_array = img_array / 255.0  # Ensure this matches your training scaling

        # Add batch dimension (1, 224, 224, 3)
        img_reshape = img_array[np.newaxis, ...]
        return img_reshape


    # Process and Predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # --- 4. Display Results (The Fixed Version) ---
    from tensorflow.keras.applications.vgg16 import decode_predictions

    # Get the top 3 predictions from the model's raw output
    # 'prediction' is the (1, 1000) array from model.predict()
    top_predictions = decode_predictions(prediction, top=3)[0]

    st.write("### Model Results:")

    # Loop through the top 3 and display them nicely
    for i, (imagenet_id, label, probability) in enumerate(top_predictions):
        # Format the label (e.g., 'golden_retriever' -> 'Golden Retriever')
        clean_label = label.replace('_', ' ').title()

        st.write(f"{i + 1}. **{clean_label}**")
        st.info(f"Confidence: {probability:.2%}")
        st.progress(float(probability))
