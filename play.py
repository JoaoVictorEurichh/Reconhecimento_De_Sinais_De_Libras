"""
Passo 3 - Reconhecimento de sinais de Libras ao vivo.

Uso:  python play.py
  q = sair

Mostra o sinal que o modelo reconhece (com voto da maioria nos ultimos
frames para estabilizar), a confianca e a probabilidade de cada classe.
"""
import json
import os
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
MIN_CONF = 0.6   # abaixo disso o modelo esta "em duvida"
model = tf.keras.models.load_model("gesture_model.keras")
class_names = json.load(open("class_names.json"))

def predict(roi):
    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(img[None].astype("float32"), verbose=0)[0]
    return class_names[int(np.argmax(probs))], float(np.max(probs)), probs

CAMERA = os.environ.get("CAMERA", "0")          # indice (0, 1...) ou URL do stream
cap = cv2.VideoCapture(int(CAMERA) if CAMERA.isdigit() else CAMERA)
if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir a camera {CAMERA!r}.")

recent = deque(maxlen=7)          # suaviza: voto da maioria nos ultimos frames

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    label, conf, probs = predict(roi)
    recent.append(label if conf >= MIN_CONF else "?")
    stable = Counter(recent).most_common(1)[0][0]

    if stable == "nada":
        message, color = "Nenhum sinal", (200, 200, 200)
    elif stable == "?":
        message, color = "Em duvida...", (0, 200, 255)
    else:
        message, color = f"Sinal: {stable}", (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), color, 3)
    cv2.putText(frame, f"Modelo ve: {label} ({conf:.0%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    # barras de probabilidade (otimo para mostrar a "incerteza" do modelo)
    for i, c in enumerate(class_names):
        bw = int(probs[i] * 150)
        cv2.rectangle(frame, (10, 50 + i * 22), (10 + bw, 64 + i * 22), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, 63 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, message, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.imshow("Sinais de Libras com CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
