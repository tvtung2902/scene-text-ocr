from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import torch
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from model.ocr_model import OCRModel
from utils.vocab import Vocab
from utils.beam import beam_search_decode
from torchvision import transforms

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
VOCAB_PATH = 'utils/vocab.yml'
YOLO_CKPT = 'checkpoint/yolo.pt'
OCR_CKPT = 'checkpoint/lstm.pt'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE= 'cpu'

vocab = Vocab(VOCAB_PATH)
vocab_size = len(vocab.char_2_idx)
yolo = YOLO(YOLO_CKPT)
ocr_model = OCRModel(vocab_size=vocab_size, hidden_size=512, n_layers=2).to(DEVICE)
checkpoint = torch.load(OCR_CKPT, map_location=DEVICE)
ocr_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
ocr_model.eval()

# Preprocess crop for OCR
def preprocess_ocr(img, target_size=(100, 420)):
    pil_img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return transform(pil_img).unsqueeze(0)

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_np = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = yolo(img_np, conf=0.5, verbose=False)
    boxes = results[0].boxes
    preds = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    keep = confs >= 0.5
    preds = preds[keep]
    confs = confs[keep]
    sorted_preds = sorted(zip(preds, confs), key=lambda item: (item[0][1], item[0][0]))

    ocr_results = []
    for box, conf in sorted_preds:
        x1, y1, x2, y2 = map(int, box)
        crop = img_np[y1:y2, x1:x2]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        crop_tensor = preprocess_ocr(crop).to(DEVICE)
        with torch.no_grad():
            output = ocr_model(crop_tensor)
            log_probs = torch.nn.functional.log_softmax(output, dim=2)
            beam_decoded = beam_search_decode(log_probs, beam_width=5, blank_index=vocab.blank_index)
            pred_texts = vocab.decode(beam_decoded)
            text = pred_texts[0]
        if text.strip():
            ocr_results.append({
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "text": text,
                "confidence": float(conf)
            })

    return JSONResponse(content={"results": ocr_results})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
