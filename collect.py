"""
Passo 1 - Coleta de dados pela webcam (10 sinais de Libras + nada).

Uso:  python collect.py
      PowerShell, DroidCam direto:  $env:CAMERA="http://IP-DO-CELULAR:4747/video"; python collect.py

IMPORTANTE: clique uma vez na JANELA DO VIDEO para ela receber as teclas
(se o terminal estiver em foco, nada acontece).

Teclas (aperte UMA vez, nao segure):
  1..9 e 0 = sinais (A B C D E F G I L V)     n = nada
  l = alterna a iluminacao atual (escuro / claro)
  Ao apertar, ha uma contagem de 3s para voce posicionar a mao, e a
  gravacao comeca sozinha. Aperte a mesma tecla de novo para pausar.
  q = sair

Cada foto e salva como <timestamp>_<iluminacao>.jpg. O timestamp vem
primeiro para que a ordem alfabetica continue sendo a ordem da coleta
(o train.py usa os ultimos 20% de cada classe como validacao).
Dica: INTERCALE as iluminacoes (ex.: 50 escuro, 50 claro, 50 escuro...).
"""
import os
import time
import cv2

SIGNS = ["A", "B", "C", "D", "E", "F", "G", "I", "L", "V"]
CLASSES = SIGNS + ["nada"]
KEYS = {ord(str((i + 1) % 10)): s for i, s in enumerate(SIGNS)}   # 1..9, 0
KEYS[ord("n")] = "nada"
LIGHTS = ["escuro", "claro"]

DATA_DIR = "data"
IMG_SIZE = 224
SAVE_EVERY_N_FRAMES = 3
MAX_PER_CLASS = 300
MAX_PER_LIGHT = MAX_PER_CLASS // len(LIGHTS)   # mantem as iluminacoes balanceadas
COUNTDOWN = 3.0
DEBOUNCE = 0.5   # ignora repeticao de tecla (segurar) por 0.5s

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)

def count(cls, light=None):
    suffix = f"_{light}.jpg" if light else ".jpg"
    return len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.endswith(suffix)])

CAMERA = os.environ.get("CAMERA", "0")          # indice (0, 1...) ou URL do stream
cap = cv2.VideoCapture(int(CAMERA) if CAMERA.isdigit() else CAMERA)
if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir a camera {CAMERA!r}. Feche outros apps que usam a camera "
                     "(Zoom, Meet...) ou defina CAMERA=1 ou CAMERA=<url do stream>.")

recording = None       # classe sendo gravada
light = LIGHTS[0]      # iluminacao atual
countdown_start = None
last_key_time = 0
frame_count = 0
WIN = "Coleta Libras"
print("Janela aberta. Clique nela e aperte 1..9/0 (sinais) ou n (nada). "
      "'l' troca a iluminacao, 'q' sai.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler a webcam.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    status, color = f"PARADO ({light}) - aperte 1..9/0 ou n", (0, 255, 0)
    if recording and countdown_start is not None:
        left = COUNTDOWN - (time.time() - countdown_start)
        if left > 0:
            status, color = f"{recording} ({light}): prepare a mao... {int(left) + 1}", (0, 255, 255)
        else:
            countdown_start = None
    if recording and countdown_start is None:
        status, color = f"GRAVANDO: {recording} ({light})", (0, 0, 255)
        frame_count += 1
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            if roi.std() < 8:
                # quadro liso (ex.: tela verde de camera virtual sem sinal): nao salva lixo
                print("Imagem vazia/lisa da camera - nada foi salvo. Verifique a camera.")
                recording = None
            elif count(recording, light) < MAX_PER_LIGHT:
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                name = f"{int(time.time() * 1000)}_{light}.jpg"
                cv2.imwrite(os.path.join(DATA_DIR, recording, name), img)
            else:
                print(f"{recording} ({light}): completo ({MAX_PER_LIGHT} fotos).")
                recording = None

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), color, 3)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for i, c in enumerate(CLASSES):
        key = "n" if c == "nada" else str((i + 1) % 10)
        txt = (f"[{key}] {c}: esc {count(c, 'escuro')} / cla {count(c, 'claro')}"
               f"  (max {MAX_PER_LIGHT} cada)")
        cv2.putText(frame, txt, (10, 60 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 255:
        continue
    if key == ord("q"):
        break
    if time.time() - last_key_time <= DEBOUNCE:
        continue
    if key == ord("l"):
        last_key_time = time.time()
        light = LIGHTS[(LIGHTS.index(light) + 1) % len(LIGHTS)]
        recording = None   # pausa para voce ajustar o fundo/luz
        print(f"Iluminacao: {light} (gravacao pausada)")
    elif key in KEYS:
        last_key_time = time.time()
        cls = KEYS[key]
        if recording == cls:
            recording = None
            print(f"Pausado: {cls}")
        else:
            recording, countdown_start = cls, time.time()
            print(f"Preparando: {cls} ({light})")
    else:
        print(f"(tecla nao reconhecida: codigo {key})")

cap.release()
cv2.destroyAllWindows()
print({c: (count(c, "escuro"), count(c, "claro")) for c in CLASSES})
