# test_cam_anti_spoof_v2.py
from pathlib import Path
import time
import numpy as np
import cv2
import torch

from antispoof_model import EffV2WithMeta, build_img_transform, IMG_SIZE, META_DIM

RUNS_DIR = Path(r"C:\Users\USUARIO\datasetproject\dataset-muestra\dataset-muestra\runs\effv2_meta")
WEIGHTS  = RUNS_DIR / "best.pt"

THRESH_LIVE = 0.70     # umbral de "live"
CAM_INDEX   = 0
USE_DSHOW   = True

# --- Detector de rostro (MediaPipe con fallback a Haar) ---
try:
    import mediapipe as mp
    mp_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    USE_MP = True
except Exception:
    USE_MP = False
    HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(frame_bgr):
    """Devuelve (x,y,w,h,conf) o None."""
    h, w = frame_bgr.shape[:2]
    if USE_MP:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = mp_detector.process(rgb)
        if not res.detections:
            return None
        d = res.detections[0]
        b = d.location_data.relative_bounding_box
        x = max(0, int(b.xmin*w)); y = max(0, int(b.ymin*h))
        ww = max(1, int(b.width*w)); hh = max(1, int(b.height*h))
        conf = float(d.score[0])
        return (x, y, ww, hh, conf)
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = HAAR.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return None
        x, y, ww, hh = faces[0]
        return (int(x), int(y), int(ww), int(hh), 0.8)

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # modelo
    model = EffV2WithMeta(pretrained=False).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    tfm = build_img_transform(IMG_SIZE)

    # cámara
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if USE_DSHOW else 0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # smoothing
    sm_win = 5
    last_probs = []

    t0 = time.time(); frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        bb = detect_face(frame)

        if bb is None:
            # sin rostro → marcamos como spoof
            prob_live = 0.0
            prob_spoof = 1.0
            prob_live_s = 0.0
            status = "SPOOF"
            color = (0, 0, 255)
        else:
            x, y, ww, hh, conf = bb
            x2, y2 = min(w, x+ww), min(h, y+hh)
            crop = frame[y:y2, x:x2]
            if crop.size == 0:
                crop = frame

            # metadata relativa
            meta = np.array([x/w, y/h, ww/w, hh/h, conf], dtype=np.float32)
            meta_t = torch.from_numpy(meta).unsqueeze(0).to(device)

            # imagen
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = tfm(rgb).unsqueeze(0).to(device)

            logits = model(img, meta_t)
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            prob_live, prob_spoof = float(prob[1]), float(prob[0])

            # suavizado
            last_probs.append(prob_live)
            if len(last_probs) > sm_win: last_probs.pop(0)
            prob_live_s = sum(last_probs)/len(last_probs)

            # estado y color
            if prob_live_s >= THRESH_LIVE:
                status = "LIVE"
                color = (0, 200, 0)
            else:
                status = "SPOOF"
                color = (0, 0, 255)

            # dibujar bbox
            cv2.rectangle(frame, (x, y), (x+ww, y+hh), color, 2)

        # porcentajes bonitos
        live_pct  = prob_live_s * 100.0
        spoof_pct = (1.0 - prob_live_s) * 100.0   # complementario suavizado

        # título grande de estado
        cv2.putText(frame, f"{status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        # detalle de probabilidades en porcentaje
        cv2.putText(frame, f"Live: {live_pct:5.1f}%   Spoof: {spoof_pct:5.1f}%",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # FPS
        frames += 1
        if frames % 10 == 0:
            fps = frames / (time.time() - t0)
        try:
            fps
        except NameError:
            fps = 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)

        cv2.imshow("Anti-Spoof — LIVE vs SPOOF (ESC para salir)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
