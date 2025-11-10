# plot_training_curves.py
from pathlib import Path
import json
import matplotlib.pyplot as plt

RUNS_DIR = Path(r"C:\Users\USUARIO\datasetproject\dataset-muestra\dataset-muestra\runs\effv2_meta")
HIST_JSON = RUNS_DIR / "history.json"
OUT_DIR = RUNS_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with HIST_JSON.open("r", encoding="utf-8") as f:
    hist = json.load(f)

epochs = [h["ep"] for h in hist]
tr_loss = [h["train"]["loss"] for h in hist]; va_loss = [h["val"]["loss"] for h in hist]
tr_acc  = [h["train"]["acc"]  for h in hist]; va_acc  = [h["val"]["acc"]  for h in hist]
tr_f1   = [h["train"]["f1"]   for h in hist]; va_f1   = [h["val"]["f1"]   for h in hist]

def save_plot(x, y1, y2, ylabel, title, fname):
    plt.figure()
    plt.plot(x, y1, label="train")
    plt.plot(x, y2, label="val")
    plt.xlabel("Época"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(OUT_DIR / fname, dpi=150)

save_plot(epochs, tr_loss, va_loss, "Loss", "Curva de pérdida", "loss_curve.png")
save_plot(epochs, tr_acc,  va_acc,  "Accuracy", "Curva de accuracy", "acc_curve.png")
save_plot(epochs, tr_f1,   va_f1,   "F1 (macro)", "Curva de F1", "f1_curve.png")

print("Gráficos guardados en:", OUT_DIR)
