# export_model_ts.py
from pathlib import Path
import torch
from antispoof_model import EffV2WithMeta, IMG_SIZE, META_DIM

RUNS_DIR = Path(r"C:\Users\USUARIO\datasetproject\dataset-muestra\dataset-muestra\runs\effv2_meta")
BEST_PT  = RUNS_DIR / "best.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EffV2WithMeta(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(BEST_PT, map_location=device))

    # TorchScript (scripted → más robusto)
    scripted = torch.jit.script(model)
    out_path = RUNS_DIR / "anti_spoof_effv2_scripted.pt"
    scripted.save(str(out_path))
    print("OK TorchScript guardado en:", out_path)

if __name__ == "__main__":
    main()
