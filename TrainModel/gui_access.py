# gui_access_pro_statedict_ux_dynamic.py
# Control de Acceso (Laboral) — Anti-Spoof + Identidad [STATE_DICT]
# - Crop cuadrado con margen (1.2x)
# - Umbral dinámico por tamaño/nitidez
# - Warm-up corto, menos latencia (frames consecutivos y cooldown)
# - Iluminación condicional
# - HUD con diagnóstico
# - Registro con doble chequeo rápido
# - UI con criterios de Nielsen (colores, feedback, prevención de errores)

from pathlib import Path
import csv, threading
from datetime import datetime

import cv2
import numpy as np
import torch
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox

# ---------- RUTAS ----------
RUNS_DIR = Path(r"C:\Users\USUARIO\datasetproject\dataset-muestra\dataset-muestra\runs\effv2_meta")
BEST_PT  = RUNS_DIR / "best.pt"

REG_DIR = RUNS_DIR / "registry"
REG_IMG_DIR = REG_DIR / "images"
LOG_CSV = RUNS_DIR / "access_log.csv"
REG_DIR.mkdir(parents=True, exist_ok=True)
REG_IMG_DIR.mkdir(parents=True, exist_ok=True)
if not LOG_CSV.exists():
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp","user","event","details"])

# ---------- MODELO (STATE_DICT) ----------
from antispoof_model import EffV2WithMeta, build_img_transform, IMG_SIZE, META_DIM
IMAGENET_TFM = build_img_transform(IMG_SIZE)

# ---------- PARÁMETROS (AJUSTADOS) ----------
THRESH_LIVE_BASE     = 0.65   # umbral base (login)
THRESH_LIVE_REG      = 0.75   # registro (ligeramente más estricto)
REQ_CONSEC_LIVE      = 20     # ~0.7 s a 30 FPS
REQ_CONSEC_LIVE_REG  = 8      # registro más ágil
WARMUP_FRAMES        = 8      # warm-up más corto
COOLDOWN_CONFIRM     = 5      # confirmación breve
SMOOTH_WIN           = 8      # ventana de suavizado
SIM_THRESHOLD        = 0.65   # identidad (coseno) mínima
MIN_FACE_REL         = 0.14   # rostro ≥14% del lado menor
MIN_LAP_VAR          = 45.0   # nitidez mínima
CAM_INDEX            = 0

def enhance_illum(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray))
    if m < 60:   gamma = 0.6
    elif m < 80: gamma = 0.8
    else:        gamma = 1.0
    if gamma != 1.0:
        inv = 1.0 / max(gamma, 1e-6)
        table = (np.linspace(0,1,256)**inv * 255).astype(np.uint8)
        bgr = cv2.LUT(bgr, table)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

# ---------- DETECTOR ----------
try:
    import mediapipe as mp
    mp_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    USE_MP = True
except Exception:
    USE_MP = False
    HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(frame_bgr):
    H, W = frame_bgr.shape[:2]
    if USE_MP:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = mp_det.process(rgb)
        if not res.detections: return None
        d = res.detections[0]
        b = d.location_data.relative_bounding_box
        x = max(0, int(b.xmin*W)); y = max(0, int(b.ymin*H))
        w = max(1, int(b.width*W)); h = max(1, int(b.height*H))
        conf = float(d.score[0])
        return (x,y,w,h,conf)
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = HAAR.detectMultiScale(gray, 1.2, 5)
        if len(faces)==0: return None
        x,y,w,h = faces[0]
        return (int(x),int(y),int(w),int(h),0.8)

# ---------- CHEQUEOS / CROP / UMBRAL DINÁMICO ----------
def face_size_ok(bbox_px, frame_hw):
    H, W = frame_hw
    x,y,w,h,_ = bbox_px
    rel = min(w/W, h/H)
    return rel >= MIN_FACE_REL

def face_sharp_enough(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return var >= MIN_LAP_VAR, var

def expand_square_bbox(bbox_px, frame_hw, scale=1.20):
    H, W = frame_hw
    x,y,w,h,_ = bbox_px
    cx, cy = x + w/2, y + h/2
    side = int(max(w, h) * scale)
    x0 = int(max(0, cx - side/2)); y0 = int(max(0, cy - side/2))
    x1 = int(min(W, x0 + side));   y1 = int(min(H, y0 + side))
    side = min(x1-x0, y1-y0)
    x1, y1 = x0 + side, y0 + side
    return x0, y0, x1, y1

def dynamic_live_threshold(face_rel, lap_var):
    t = THRESH_LIVE_BASE
    if face_rel >= 0.22 and lap_var >= 120:   t -= 0.05        # grande + muy nítido
    elif face_rel <= 0.12 or lap_var <= 35:   t += 0.08        # pequeño o suave
    return float(np.clip(t, 0.55, 0.85))

# ---------- CLASE ANTI-SPOOF (STATE_DICT) ----------
class AntiSpoofSD:
    def __init__(self, weights_path: Path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = EffV2WithMeta(pretrained=False).to(self.device).eval()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
    @torch.no_grad()
    def predict_live(self, crop_bgr, bbox_px, frame_hw):
        H, W = frame_hw
        x,y,w,h,conf = bbox_px
        meta = np.array([x/W, y/H, w/W, h/H, float(conf)], dtype=np.float32)[None, :]
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        imgT = IMAGENET_TFM(rgb).unsqueeze(0).to(self.device)
        metaT= torch.from_numpy(meta).to(self.device)
        logits = self.model(imgT, metaT)            # [B,2] -> [spoof, live]
        prob   = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return float(prob[1])                       # prob_live

# ---------- IDENTIDAD ----------
from facenet_pytorch import InceptionResnetV1, MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn  = MTCNN(image_size=160, margin=20, keep_all=False, device=device)

def face_embedding(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = mtcnn(rgb)
    if img is None: return None
    with torch.no_grad():
        emb = resnet(img.unsqueeze(0).to(device)).cpu().numpy()[0]
    return emb / (np.linalg.norm(emb) + 1e-9)

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def save_user_embedding(username, embeddings):
    emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    np.save(REG_DIR / f"{username}.npy", emb)

def load_users():
    return {p.stem: np.load(p) for p in REG_DIR.glob("*.npy")}

def log_event(user, event, details=""):
    with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), user, event, details])

# ---------- THEME / UX ----------
PALETTE = {
    "bg": "#0f172a", "card": "#111827", "muted": "#94a3b8", "text": "#e5e7eb",
    "primary": "#2563eb", "success": "#10b981", "danger": "#ef4444", "warn": "#f59e0b"
}
def style_init(root: tk.Tk):
    root.configure(bg=PALETTE["bg"])
    style = ttk.Style(root)
    try: style.theme_use("clam")
    except: pass
    style.configure("TNotebook", background=PALETTE["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", background=PALETTE["card"], foreground=PALETTE["text"], padding=(14, 8))
    style.map("TNotebook.Tab", background=[("selected", PALETTE["primary"])], foreground=[("selected", "#fff")])
    style.configure("Card.TFrame", background=PALETTE["card"])
    style.configure("TFrame", background=PALETTE["bg"])
    style.configure("TLabel", background=PALETTE["bg"], foreground=PALETTE["text"])
    style.configure("Muted.TLabel", background=PALETTE["bg"], foreground=PALETTE["muted"])
    style.configure("H1.TLabel", font=("Segoe UI", 14, "bold"), foreground=PALETTE["text"], background=PALETTE["bg"])
    style.configure("TEntry", fieldbackground="#0b1220", foreground=PALETTE["text"])
    style.configure("Primary.TButton", background=PALETTE["primary"], foreground="#fff")
    style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#1e293b")])
    style.configure("Success.TButton", background=PALETTE["success"], foreground="#06281f")
    style.map("Success.TButton", background=[("active", "#059669"), ("disabled", "#1e293b")])
    style.configure("Danger.TButton", background=PALETTE["danger"], foreground="#fff")
    style.map("Danger.TButton", background=[("active", "#dc2626"), ("disabled", "#1e293b")])

class Tooltip:
    def __init__(self, widget, text, delay=500):
        self.widget, self.text, self.delay = widget, text, delay
        self.id = None; self.tw = None
        widget.bind("<Enter>", self.on_enter); widget.bind("<Leave>", self.on_leave)
    def on_enter(self, _): self.schedule()
    def on_leave(self, _): self.unschedule(); self.hidetip()
    def schedule(self):
        self.unschedule(); self.id = self.widget.after(self.delay, self.showtip)
    def unschedule(self):
        _id, self.id = self.id, None
        if _id: self.widget.after_cancel(_id)
    def showtip(self):
        if self.tw: return
        x = self.widget.winfo_rootx() + 20; y = self.widget.winfo_rooty() + 25
        self.tw = tk.Toplevel(self.widget); self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tw, text=self.text, justify="left", background="#111827",
                       foreground="#e5e7eb", relief="solid", borderwidth=1,
                       font=("Segoe UI", 9), padx=8, pady=6)
        lbl.pack()
    def hidetip(self):
        tw, self.tw = self.tw, None
        if tw: tw.destroy()

# ---------- APP ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        style_init(self)
        self.title("Control de Acceso — Anti-Spoof + Identidad [STATE_DICT]")
        self.geometry("980x720"); self.minsize(920, 660)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # estado
        self.cap = None
        self.running = False
        self.th = None
        self.last_live = []
        self.consec_live = 0
        self.status = tk.StringVar(value="Consejo: ESC para cancelar, ESPACIO para foto en Registro.")

        # estado registro
        self.reg_mode = False
        self.reg_username = tk.StringVar(value="")
        self.reg_target = tk.IntVar(value=5)
        self.reg_capt = 0
        self.reg_kept_embeddings = []
        self.reg_last_live = []
        self._want_shutter = False

        if not BEST_PT.exists():
            messagebox.showerror("Modelo", f"No se encontró best.pt en:\n{BEST_PT}")
            self.destroy(); return
        self.antispoof = AntiSpoofSD(BEST_PT)

        # Encabezado
        top = ttk.Frame(self, style="TFrame"); top.pack(fill="x", padx=12, pady=(10, 0))
        ttk.Label(top, text="Control de Acceso (Laboral)", style="H1.TLabel").pack(side="left")
        self.lbl_state = ttk.Label(top, text="Listo", style="Muted.TLabel"); self.lbl_state.pack(side="right")

        # Notebook
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=12, pady=12)
        self.tab_access = ttk.Frame(nb, style="Card.TFrame"); nb.add(self.tab_access, text="Acceso")
        self.tab_register = ttk.Frame(nb, style="Card.TFrame"); nb.add(self.tab_register, text="Registro")
        self.tab_log = ttk.Frame(nb, style="Card.TFrame"); nb.add(self.tab_log, text="Historial")
        self.tab_cfg = ttk.Frame(nb, style="Card.TFrame"); nb.add(self.tab_cfg, text="Config")

        self.build_tab_access()
        self.build_tab_register()
        self.build_tab_log()
        self.build_tab_cfg()

        # barra inferior
        statusbar = ttk.Label(self, textvariable=self.status, anchor="w", style="Muted.TLabel")
        statusbar.pack(side="bottom", fill="x", padx=12, pady=(0, 10))

        self.bind_all("<space>", lambda e: self.trigger_shutter())

    # ---------- UI: ACCESO ----------
    def build_tab_access(self):
        head = ttk.Frame(self.tab_access, style="Card.TFrame"); head.pack(fill="x", pady=(10, 0))
        self.lbl_status = ttk.Label(head, text="Estado: Listo", font=("Segoe UI", 12), style="TLabel")
        self.lbl_status.pack(side="left", padx=10, pady=8)
        btns = ttk.Frame(self.tab_access, style="Card.TFrame"); btns.pack(fill="x")
        self.btn_login = ttk.Button(btns, text="Iniciar Login", style="Primary.TButton", command=self.start_login)
        self.btn_login.pack(side="left", padx=10, pady=8); Tooltip(self.btn_login, "Verificación de vida + identidad")
        self.btn_stop = ttk.Button(btns, text="Detener", style="Danger.TButton", command=self.stop_cam, state="disabled")
        self.btn_stop.pack(side="left", padx=10, pady=8); Tooltip(self.btn_stop, "Detiene la cámara")

        frame = ttk.Frame(self.tab_access, style="Card.TFrame"); frame.pack(pady=6)
        self.canvas = tk.Canvas(frame, width=860, height=484, bg="#0b1220", highlightthickness=1, highlightbackground="#1f2937")
        self.canvas.pack()

    # ---------- UI: REGISTRO ----------
    def build_tab_register(self):
        row = ttk.Frame(self.tab_register, style="Card.TFrame"); row.pack(fill="x", pady=(10, 0))
        ttk.Label(row, text="Usuario:", style="TLabel").pack(side="left", padx=(12, 4))
        ent_user = ttk.Entry(row, textvariable=self.reg_username, width=24); ent_user.pack(side="left", padx=4)
        Tooltip(ent_user, "Identificador (p. ej., jsaavedra)")
        ttk.Label(row, text="Capturas:", style="TLabel").pack(side="left", padx=(16, 4))
        spn = ttk.Spinbox(row, textvariable=self.reg_target, from_=3, to=10, width=5); spn.pack(side="left", padx=4)
        Tooltip(spn, "Fotos necesarias (recomendado 5)")

        btns = ttk.Frame(self.tab_register, style="Card.TFrame"); btns.pack(fill="x", pady=6)
        self.btn_reg_start = ttk.Button(btns, text="Iniciar Registro", style="Primary.TButton", command=self.start_register)
        self.btn_reg_start.pack(side="left", padx=10, pady=6)
        self.btn_reg_shot = ttk.Button(btns, text="Tomar foto (ESPACIO)", style="Success.TButton",
                                       command=self.trigger_shutter, state="disabled")
        self.btn_reg_shot.pack(side="left", padx=10, pady=6)
        self.btn_reg_end = ttk.Button(btns, text="Finalizar", style="Danger.TButton", command=self.finish_register, state="disabled")
        self.btn_reg_end.pack(side="left", padx=10, pady=6)

        self.reg_info = ttk.Label(self.tab_register, text="Listo para registrar.", style="Muted.TLabel"); self.reg_info.pack(pady=(2, 8))
        frame = ttk.Frame(self.tab_register, style="Card.TFrame"); frame.pack(pady=6)
        self.reg_canvas = tk.Canvas(frame, width=860, height=484, bg="#0b1220", highlightthickness=1, highlightbackground="#1f2937")
        self.reg_canvas.pack()

    # ---------- UI: HISTORIAL ----------
    def build_tab_log(self):
        frm = ttk.Frame(self.tab_log, style="Card.TFrame"); frm.pack(fill="both", expand=True, padx=8, pady=8)
        cols = ("ts", "user", "event", "details")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings")
        for cid, title, width in [("ts","Fecha/Hora",220),("user","Usuario",160),("event","Evento",160),("details","Detalles",320)]:
            self.tree.heading(cid, text=title); self.tree.column(cid, width=width, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(frm, style="Card.TFrame"); side.pack(side="left", fill="y", padx=(8,0))
        b = ttk.Button(side, text="Refrescar", style="Primary.TButton", command=self.load_log); b.pack(pady=(0,6), fill="x")
        self.load_log()

    # ---------- UI: CONFIG ----------
    def build_tab_cfg(self):
        cfg = ttk.Frame(self.tab_cfg, style="Card.TFrame"); cfg.pack(pady=14, padx=12, fill="x")
        ttk.Label(cfg, text="Umbral base Live (login):", style="TLabel").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.var_thr_login = tk.DoubleVar(value=THRESH_LIVE_BASE)
        ent_thr = ttk.Entry(cfg, textvariable=self.var_thr_login, width=6); ent_thr.grid(row=0, column=1, sticky="w")
        ttk.Label(cfg, text="Frames LIVE consecutivos (login):", style="TLabel").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.var_req_login = tk.IntVar(value=REQ_CONSEC_LIVE)
        ent_req = ttk.Entry(cfg, textvariable=self.var_req_login, width=6); ent_req.grid(row=1, column=1, sticky="w")
        ttk.Label(cfg, text="Nota: Registro usa umbral propio y doble verificación.", style="Muted.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=10)

    # ---------- Cámara ----------
    def open_cam(self):
        if self.cap and self.cap.isOpened(): return
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        try: self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception: pass

    def stop_cam(self):
        self.running = False
        if self.th and self.th.is_alive(): self.th.join(timeout=1.0)
        if self.cap: self.cap.release(); self.cap = None
        cv2.destroyAllWindows()
        self.lbl_status.config(text="Estado: Detenido"); self.lbl_state.config(text="Cámara detenida")
        if hasattr(self, "btn_stop"): self.btn_stop.configure(state="disabled")
        if hasattr(self, "btn_login"): self.btn_login.configure(state="normal")
        if hasattr(self, "btn_reg_start"): self.btn_reg_start.configure(state="normal")
        if hasattr(self, "btn_reg_shot"): self.btn_reg_shot.configure(state="disabled")
        if hasattr(self, "btn_reg_end"): self.btn_reg_end.configure(state="disabled")
        self.reg_mode = False

    # ==================== LOGIN ====================
    def start_login(self):
        if self.running: return
        users = load_users()
        if not users:
            messagebox.showwarning("Login", "No hay usuarios registrados."); return
        self.open_cam()
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Cámara", "No se pudo abrir la cámara."); return
        self.running = True
        self.last_live = []; self.consec_live = 0
        self.lbl_status.config(text="Estado: Login en progreso… mire a la cámara.")
        self.lbl_state.config(text="Login: activo")
        self.btn_login.configure(state="disabled"); self.btn_stop.configure(state="normal")
        self.th = threading.Thread(target=self._loop_login, daemon=True); self.th.start()

    def _loop_login(self):
        users = load_users()
        warm = 0
        live_buffer = []
        confirm_count = 0

        while self.running:
            ok, frame = self.cap.read()
            if not ok: break
            H, W = frame.shape[:2]
            status = "SPOOF/NO-LIVE"; color = (239,68,68)
            live_s, lapv, face_rel, thr_live = 0.0, 0.0, 0.0, float(self.var_thr_login.get())

            bb = detect_face(frame)
            if bb is not None:
                x,y,w,h,conf = bb
                # Crop cuadrado con margen
                x0,y0,x1,y1 = expand_square_bbox(bb, (H,W), scale=1.20)
                face = frame[y0:y1, x0:x1]
                if face.size > 0:
                    face_rel = min((x1-x0)/W, (y1-y0)/H)
                    sharp_ok, lapv = face_sharp_enough(face)
                    thr_live = dynamic_live_threshold(face_rel, lapv)
                    face_in = enhance_illum(face) if float(np.mean(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))) < 80 else face

                    live = self.antispoof.predict_live(face_in, (x,y,w,h,conf), (H,W))
                    live_buffer.append(live)
                    if len(live_buffer) > SMOOTH_WIN: live_buffer.pop(0)
                    live_s = sum(live_buffer)/len(live_buffer)

                    if warm < WARMUP_FRAMES:
                        warm += 1
                        status = "Inicializando…"; color = (245,158,11)
                    else:
                        req_login = int(self.var_req_login.get())
                        if live_s >= thr_live and sharp_ok and face_rel >= MIN_FACE_REL:
                            self.consec_live += 1
                        else:
                            self.consec_live = 0; confirm_count = 0

                        color = (16,185,129) if self.consec_live >= req_login else (239,68,68)

                        if self.consec_live >= req_login:
                            confirm_count += 1
                            if confirm_count >= COOLDOWN_CONFIRM:
                                emb = face_embedding(face)  # embedding del crop cuadrado
                                if emb is not None:
                                    best_user, best_sim = None, -1.0
                                    for u, ref in users.items():
                                        sim = cos_sim(emb, ref)
                                        if sim > best_sim: best_sim, best_user = sim, u
                                    if best_sim >= SIM_THRESHOLD:
                                        status = f"ACCESO: {best_user}"; color = (16,185,129)
                                        log_event(best_user, "login_success", f"sim={best_sim:.3f}, lapv={lapv:.1f}")
                                        self.running = False
                                    else:
                                        status = "IDENTIDAD NO COINCIDE"; color = (239,68,68); confirm_count = 0
                                else:
                                    status = "NO FACE EMB"; color = (239,68,68); confirm_count = 0

                # dibuja bbox original y del crop (referencia)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.rectangle(frame, (x0,y0), (x1,y1), (102,102,255), 1)

            # HUD
            cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
            cv2.putText(frame,
                f"Live {live_s*100:5.1f}%  Thr:{thr_live:.2f}  Lap:{lapv:4.0f}  Size:{face_rel*100:4.0f}%  Warm:{warm}/{WARMUP_FRAMES}  Cons:{self.consec_live}/{int(self.var_req_login.get())}",
                (12,62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245,245,245), 2, cv2.LINE_AA)

            cv2.imshow("Acceso — ESC para cancelar", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False; log_event("-", "login_cancel", ""); break

            if not self.running:
                self.lbl_status.config(text=status); self.lbl_state.config(text="Login: finalizado")
                self.load_log(); break

        self.stop_cam()

    # ==================== REGISTRO ====================
    def start_register(self):
        if self.running: return
        username = self.reg_username.get().strip()
        if not username:
            messagebox.showwarning("Registro", "Ingrese un usuario."); return
        if (REG_DIR / f"{username}.npy").exists():
            if not messagebox.askyesno("Registro", f"Ya existe '{username}'. ¿Sobrescribir?"):
                return

        target = int(self.reg_target.get());
        if target < 3: target = 3

        self.reg_capt = 0; self.reg_kept_embeddings = []; self.reg_last_live = []
        self.reg_mode = True
        self.reg_info.config(text=f"Registrando a '{username}'. Capturas 0/{target} — Umbral LIVE {THRESH_LIVE_REG:.2f}")

        self.open_cam()
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Cámara", "No se pudo abrir la cámara.")
            self.reg_mode = False; return

        self.btn_reg_start.configure(state="disabled"); self.btn_reg_shot.configure(state="normal")
        self.btn_reg_end.configure(state="normal"); self.btn_login.configure(state="disabled"); self.btn_stop.configure(state="normal")

        self.running = True
        self.th = threading.Thread(target=self._loop_register, daemon=True); self.th.start()

    def finish_register(self):
        if self.reg_mode and len(self.reg_kept_embeddings) >= max(3, int(self.reg_target.get())//2):
            username = self.reg_username.get().strip()
            save_user_embedding(username, self.reg_kept_embeddings)
            log_event(username, "register", f"shots={len(self.reg_kept_embeddings)}")
            messagebox.showinfo("Registro", f"{username} registrado con {len(self.reg_kept_embeddings)} capturas.")
            self.load_log()
        elif self.reg_mode:
            messagebox.showwarning("Registro", "No se alcanzaron capturas LIVE suficientes.")
        self.reg_mode = False
        self.stop_cam()
        self.reg_info.config(text="Listo para registrar.")
        self.reg_capt = 0; self.reg_kept_embeddings = []; self.reg_last_live = []
        self.btn_reg_start.configure(state="normal")

    def trigger_shutter(self):
        if self.reg_mode:
            self._want_shutter = True
            self.status.set("Intentando capturar… (necesita LIVE ≥ umbral)")
        else:
            self.status.set("Inicie registro antes de capturar.")

    def _loop_register(self):
        username = self.reg_username.get().strip()
        target = int(self.reg_target.get())
        flash_frames = 0

        while self.running and self.reg_mode:
            ok, frame = self.cap.read()
            if not ok: break
            H, W = frame.shape[:2]
            live_s, lapv, face_rel = 0.0, 0.0, 0.0
            bb = detect_face(frame)
            if bb:
                x,y,w,h,conf = bb
                x0,y0,x1,y1 = expand_square_bbox(bb, (H,W), scale=1.20)
                face = frame[y0:y1, x0:x1]
                if face.size > 0:
                    face_rel = min((x1-x0)/W, (y1-y0)/H)
                    sharp_ok, lapv = face_sharp_enough(face)
                    thr_live = max(THRESH_LIVE_REG, dynamic_live_threshold(face_rel, lapv))
                    face_in = enhance_illum(face) if float(np.mean(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))) < 80 else face

                    live = self.antispoof.predict_live(face_in, (x,y,w,h,conf), (H,W))
                    self.reg_last_live.append(live)
                    if len(self.reg_last_live) > SMOOTH_WIN: self.reg_last_live.pop(0)
                    live_s = sum(self.reg_last_live)/len(self.reg_last_live)
                    color = (16,185,129) if live_s >= thr_live and sharp_ok and face_rel >= MIN_FACE_REL else (239,68,68)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                    cv2.rectangle(frame, (x0,y0), (x1,y1), (102,102,255), 1)

                    # Captura manual con doble verificación rápida
                    if self._want_shutter and color == (16,185,129):
                        ok2 = False
                        for _ in range(1):  # 1 frame extra
                            okr, fr2 = self.cap.read()
                            if not okr: break
                            bb2 = detect_face(fr2)
                            if not bb2: break
                            X,Y,W2,H2,c2 = bb2
                            x0b,y0b,x1b,y1b = expand_square_bbox(bb2, fr2.shape[:2], scale=1.20)
                            f2 = fr2[y0b:y1b, x0b:x1b]
                            if f2.size == 0: break
                            use2 = enhance_illum(f2) if float(np.mean(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))) < 80 else f2
                            if self.antispoof.predict_live(use2, (X,Y,W2,H2,c2), fr2.shape[:2]) >= THRESH_LIVE_REG:
                                ok2 = True; break
                        if ok2:
                            emb = face_embedding(face)
                            if emb is not None:
                                self.reg_kept_embeddings.append(emb)
                                self.reg_capt += 1
                                (REG_IMG_DIR / username).mkdir(parents=True, exist_ok=True)
                                fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                                cv2.imwrite(str((REG_IMG_DIR / username / fname)), face)
                                flash_frames = 3
                                self.status.set(f"Foto guardada ({self.reg_capt}/{target}).")
                            else:
                                self.status.set("No se pudo extraer el rostro para embedding.")
                        self._want_shutter = False

            if flash_frames > 0:
                cv2.rectangle(frame, (0,0), (W-1,H-1), (255,255,255), 20); flash_frames -= 1

            cv2.putText(frame, f"Usuario: {username}  Capturas {self.reg_capt}/{target}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Live {live_s*100:5.1f}% Thr:{max(THRESH_LIVE_REG, dynamic_live_threshold(face_rel, lapv)):.2f} "
                               f"Lap:{lapv:4.0f} Size:{face_rel*100:4.0f}% (ESPACIO = Foto)",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2, cv2.LINE_AA)

            cv2.imshow("Registro — ESC = Finalizar, ESPACIO = Foto", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == 32: self._want_shutter = True

            self.reg_info.config(text=f"Registrando '{username}': {self.reg_capt}/{target} válidas (LIVE)")
            if self.reg_capt >= target: break

        self.finish_register()

    # ---------- utilidades ----------
    def load_log(self):
        if hasattr(self, "tree"):
            for i in self.tree.get_children(): self.tree.delete(i)
            if LOG_CSV.exists():
                with LOG_CSV.open("r", encoding="utf-8") as f:
                    rd = csv.reader(f); rows = list(rd)[1:]
                for ts,user,ev,det in rows[-500:]:
                    self.tree.insert("", "end", values=(ts,user,ev,det))

    def on_close(self):
        self.reg_mode = False
        self.stop_cam()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
