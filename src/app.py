import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'thai_char_model.keras')
LABEL_PATH = os.path.join(os.path.dirname(__file__), 'label_classes.joblib')
CSV_PATH = os.path.join(os.path.dirname(__file__), '../datasets/master_file.csv')

model = tf.keras.models.load_model(MODEL_PATH)
class_names = joblib.load(LABEL_PATH)

thai_char_map = {}
try:
    with open(CSV_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            thai_char_map[row['label']] = row['original_character']
except Exception as e:
    print(f"คำเตือน: อ่านไฟล์ master_file.csv ไม่สำเร็จ ({e}) จะแสดงผลเป็นชื่อคลาสภาษาอังกฤษแทน")

def predict_image(img):
    if img is None:
        return "กรุณาอัปโหลดรูปภาพ"
    
    img_pil = Image.fromarray(img).convert('L')
    img_pil = img_pil.resize((64, 64))
    
    img_array = np.array(img_pil) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    
    folder_name = class_names[predicted_index]
    
    thai_char = thai_char_map.get(folder_name, folder_name)
    
    return thai_char

app = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="อัปโหลดรูปภาพตัวอักษรไทย"),
    outputs=gr.Textbox(label="ผลการทำนาย (ตัวอักษร)", show_copy_button=True),
    title="Thai Character Classification",
    description="อัปโหลดรูปภาพตัวอักษรไทย สระ วรรณยุกต์ หรือตัวเลข เพื่อให้ AI ทำนายผล",
    flagging_mode="never"
)

if __name__ == "__main__":
    app.launch()