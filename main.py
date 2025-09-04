from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import fashion_mnist

# Load dataset
(_, _), (X_test, y_test) = fashion_mnist.load_data()

# Pick one test sample (e.g., index 0)
idx = 0
image = X_test[idx]
label = y_test[idx]

# Save as PNG
img = Image.fromarray(image)
img.save("sample.png")

print(f"Saved sample.png with true label: {label}")

model = tf.keras.models.load_model("fashion_model_adam_with_lrd.h5")

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Create FastAPI app
app = FastAPI(title="Fashion-MNIST FastAPI Service")

# Input schema
class Item(BaseModel):
    data: list  # expects 784 pixel values (flattened 28x28 image)

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Fashion-MNIST FastAPI service!"}

# Prediction endpoint
@app.post("/predict")
def predict(item: Item):
    # Convert input to numpy array and preprocess
    x = np.array(item.data).reshape(1, 28, 28) / 255.0

    # Predict with the model
    preds = model.predict(x)
    class_index = int(np.argmax(preds))
    class_name = class_names[class_index]

    return {
        "prediction_index": class_index,
        "prediction_label": class_name
    }

# Prediction from image upload
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Open the uploaded image, force grayscale, resize to 28x28
        image = Image.open(file.file).convert("L")
        image = image.resize((28, 28))

        # Flatten into 784 vector (since your model expects 784 inputs)
        x = np.array(image).reshape(1, 784) / 255.0

        # Predict
        preds = model.predict(x)
        class_index = int(np.argmax(preds))

        return {
            "prediction_index": class_index,
            "prediction_label": class_names[class_index]
        }

    except Exception as e:
        return {"error": str(e)}

