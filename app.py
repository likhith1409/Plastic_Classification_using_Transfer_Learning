import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


st.set_page_config(
    page_title="Plastic Classification",
    page_icon="🌍",
    layout="wide"
)

# Load the saved model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plastic_classifier.h5')
    return model

model = load_model()

# Class names (update according to your model)
CLASS_NAMES = ["HDPE", "LDPE", "Other", "PET", "PP", "PS", "PVC"]

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img, dtype=np.float32)  # Convert to float32
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array


# Function to make predictions
def predict(image_file):
    img = Image.open(image_file)
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return img, predicted_class, confidence

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ("Home", "Model Training"))

if page == "Home":
    # Home page content
    st.title("🌍 Plastic Classification")
    st.markdown("Upload an image of plastic, and the model will predict its type.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of plastic for classification."
    )

    # Display results
    if uploaded_file is not None:
        # Make prediction
        img, predicted_class, confidence = predict(uploaded_file)

        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Display prediction results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", predicted_class)
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        # Show a bar chart of predictions
        st.subheader("Class Probabilities")
        predictions = model.predict(preprocess_image(img))[0]
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, predictions, color='#4CAF50')
        ax.set_xlabel("Plastic Type")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

elif page == "Model Training":
    # Model training page content
    st.title("Model Training Overview")

    st.markdown("""
    This section explains how the plastic classification model was trained, and provides detailed insights into the training process, performance, and results.

    ## Dataset Overview
    The dataset used to train this model consists of images of plastic items from seven different classes:
    - **HDPE** (High-Density Polyethylene)
    - **LDPE** (Low-Density Polyethylene)
    - **Other**
    - **PET** (Polyethylene Terephthalate)
    - **PP** (Polypropylene)
    - **PS** (Polystyrene)
    - **PVC** (Polyvinyl Chloride)

    The dataset contains a total of 1,811 images split into training, validation, and test sets:
    - **Training set**: 1,270 images
    - **Validation set**: 354 images
    - **Test set**: 187 images

    ## Preprocessing and Augmentation
    We used several preprocessing techniques to prepare the images for training:
    - **Resizing**: All images were resized to a uniform size of 224x224 pixels.
    - **Normalization**: Pixel values were scaled to the range [0, 1] by dividing by 255.
    
    **Data Augmentation** was applied to the training set to improve the model's ability to generalize:
    - Random rotation, width/height shifts, shearing, zooming, and horizontal flipping.

    ## Model Architecture
    We utilized **MobileNetV2**, a lightweight and efficient pre-trained convolutional neural network model as the base for transfer learning. The base model was frozen (non-trainable) and new dense layers were added on top for the classification task:
    
    - **Base Model**: MobileNetV2 (pre-trained on ImageNet)
    - **Additional Layers**:
        - **GlobalAveragePooling2D**: To reduce the dimensionality.
        - **Dropout (0.2)**: To reduce overfitting.
        - **Dense (128 units)**: Fully connected layer with ReLU activation.
        - **Dense (7 units)**: Output layer with softmax activation for multi-class classification.

    ## Training Process
    The model was trained using the **Adam optimizer** with an exponential learning rate decay:
    - **Initial learning rate**: 1e-4
    - **Decay steps**: 100
    - **Decay rate**: 0.96

    The model was trained for 50 epochs, and early stopping was used to prevent overfitting. The best model was saved during training based on the highest validation accuracy.

    ## Model Results
    After training, the model achieved the following results on the test set:
    - **Test accuracy**: 75.94%
    - **Test loss**: 0.6468

    """)

    # Display the model summary
    st.subheader("Model Architecture")
    st.text("""
    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
     mobilenetv2_1.00_224 (Funct  (None, 7, 7, 1280)       2257984   
     ional)                                                          
                                                                 
     global_average_pooling2d (G  (None, 1280)             0         
     lobalAveragePooling2D)                                          
                                                                 
     dropout (Dropout)           (None, 1280)              0         
                                                                 
     dense (Dense)               (None, 128)               163968    
                                                                 
     dense_1 (Dense)             (None, 7)                 903       
                                                                 
    =================================================================
    Total params: 2,422,855
    Trainable params: 164,871
    Non-trainable params: 2,257,984
    _________________________________________________________________
    """)

    # Upload and display the graph images (Accuracy/Loss and Confusion Matrix)
    st.subheader("Training Accuracy & Loss Graph")
    st.image("assets\graph1.png", caption="Training Accuracy and Loss")

    st.subheader("Confusion Matrix")
    st.image("assets\graph2.png", caption="Confusion Matrix")

    # Conclusion and observations
    st.markdown("""
    ## Conclusion
    The model achieved an accuracy of **75.94%** after training. This indicates that the model is able to accurately classify plastic types most of the time, although there is still room for improvement.

    The confusion matrix and accuracy/loss graphs provide a detailed visualization of the model's performance. The confusion matrix highlights which plastic types the model confuses most often, and the training graphs show how the model's accuracy improved over time.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    **Plastic Classification App**  
    Built with Streamlit and TensorFlow.  
    Model trained to classify plastic types: HDPE, LDPE, Other, PET, PP, PS, PVC.
    """
)
