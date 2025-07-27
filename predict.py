import sys
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
from model.ocr_model import OCRModel
from utils.vocab import Vocab
from utils.beam import beam_search_decode

def load_ocr_model(ocr_ckpt, vocab_size, hidden_size=512, n_layers=2, device='cpu'):
    model = OCRModel(vocab_size=vocab_size, hidden_size=hidden_size, n_layers=n_layers)
    checkpoint = torch.load(ocr_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def preprocess_ocr(img, target_size=(100, 420)):
    pil_img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return tensor

def main(img_path):
    yolo_ckpt = 'checkpoint/yolo.pt'
    ocr_ckpt = 'checkpoint/lstm.pt'
    vocab_yml = 'utils/vocab.yml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab = Vocab(vocab_yml)
    vocab_size = len(vocab.char_2_idx)

    yolo = YOLO(yolo_ckpt)
    ocr = load_ocr_model(ocr_ckpt, vocab_size, device=device)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo(img_path, conf=0.5, verbose=False)
    boxes = results[0].boxes
    preds = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    conf_thres = 0.5
    keep = confs >= conf_thres
    preds = preds[keep]
    confs = confs[keep]

    ocr_results = []
    for box, conf in zip(preds, confs):
        x1, y1, x2, y2 = map(int, box)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        crop_tensor = preprocess_ocr(crop).to(device)
        with torch.no_grad():
            output = ocr(crop_tensor)
            log_probs = torch.nn.functional.log_softmax(output, dim=2)
            beam_decoded = beam_search_decode(log_probs, beam_width=5, blank_index=vocab.blank_index)
            pred_texts = vocab.decode(beam_decoded)
            text = pred_texts[0]
        ocr_results.append({'box': (x1, y1, x2, y2), 'text': text, 'conf': conf})

    vis_img = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    for res in ocr_results:
        x1, y1, x2, y2 = res['box']
        text = res['text']
        conf = res['conf']
        if text.strip() == '':
            continue
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = x1 + 2
        text_y = max(th + 4, y1 - 6)
        cv2.putText(vis_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('YOLO + OCR result')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = 'image-3.png'
    main(img_path)
