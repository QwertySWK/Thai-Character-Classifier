import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'thai_char_model.keras')
LABEL_PATH = os.path.join(os.path.dirname(__file__), 'label_classes.joblib')

model = tf.keras.models.load_model(MODEL_PATH)
class_names = joblib.load(LABEL_PATH)

def predict_image(img):
    if img is None:
        return "กรุณาอัปโหลดรูปภาพ"
    
    img_pil = Image.fromarray(img).convert('L')
    img_pil = img_pil.resize((64, 64))
    
    img_array = np.array(img_pil) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    
    return class_names[predicted_index]

app = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="อัปโหลดรูปภาพตัวอักษรไทย"),
    outputs=gr.Textbox(label="ผลการทำนาย (ตัวอักษร)"),
    title="Thai Character Classification",
    description="อัปโหลดรูปภาพตัวอักษรไทย สระ วรรณยุกต์ หรือตัวเลข เพื่อให้ AI ทำนาย (รองรับ 95 คลาส)",
    flagging_mode="never"
)

if __name__ == "__main__":
    app.launch()