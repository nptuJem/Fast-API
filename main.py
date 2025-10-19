from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# 載入模型
model = load_model("efficientnetb3_multilabel_final.keras")
labels = ['pharyngitis', 'pneumonia']  # 你可以改成你模型的症狀列表

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 讀取圖片
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224,224))  # 調整大小，依模型輸入
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)

    # 模型預測
    preds = model.predict(x)[0]
    preds_dict = {label: float(pred) for label, pred in zip(labels, preds)}
    return JSONResponse(content={"predictions": preds_dict})
