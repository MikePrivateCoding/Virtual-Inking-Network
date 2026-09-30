from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CODE_DIR = BASE_DIR / "code"
INPUT_NPY_DIR = BASE_DIR / "inputs_4096_npy"
MODEL_PATH = BASE_DIR / "model" / "best_classifier.pth"
OUT_DIR = BASE_DIR / "demo_output"

sys.path.insert(0, str(CODE_DIR))

from init import init_parameters
from networks import Classifier

REGION_IDS = ["region_1", "region_2"]
PATCH_GRID = 16
FEATURE_DIM = 384
PATCH_TILE_SIZE = 32
FINAL_SIZE = 4096
PROB_RED_THRESHOLD = 0.50
RED = np.array([255, 0, 0], dtype=np.uint8)
BLUE = np.array([0, 0, 255], dtype=np.uint8)


def load_model(device):
    config = init_parameters(is_training=False)
    model = Classifier(config)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_features(region_id):
    path = INPUT_NPY_DIR / f"{region_id}.npy"
    features = np.load(path).astype(np.float32)
    expected_shape = (PATCH_GRID * PATCH_GRID, FEATURE_DIM)
    if features.shape != expected_shape:
        raise ValueError(f"{region_id}: expected {expected_shape}, got {features.shape}")
    return features


def predict_labels(model, features, device):
    x = torch.from_numpy(features).float().reshape(features.shape[0], 1, -1).to(device)
    with torch.no_grad():
        outputs = model(x).squeeze(1)
        probs = F.softmax(outputs, dim=1)
        labels = (probs[:, 1] >= PROB_RED_THRESHOLD).long()
    return labels.cpu().numpy()


def make_concat_image(labels):
    canvas = Image.new("RGB", (PATCH_GRID * PATCH_TILE_SIZE, PATCH_GRID * PATCH_TILE_SIZE))
    for patch_idx, label in enumerate(labels):
        row = patch_idx // PATCH_GRID
        col = patch_idx % PATCH_GRID
        color = RED if label == 1 else BLUE
        tile_array = np.tile(color.reshape(1, 1, 3), (PATCH_TILE_SIZE, PATCH_TILE_SIZE, 1))
        canvas.paste(Image.fromarray(tile_array), (col * PATCH_TILE_SIZE, row * PATCH_TILE_SIZE))
    return canvas


def make_final_image(concat_image):
    small = concat_image.resize((32, 32), Image.NEAREST)
    return small.resize((FINAL_SIZE, FINAL_SIZE), Image.NEAREST)


def process_region(region_id, model, device):
    features = load_features(region_id)
    labels = predict_labels(model, features, device)
    final_image = make_final_image(make_concat_image(labels))
    out_path = OUT_DIR / f"{region_id}_rgb_output_4096.png"
    final_image.save(out_path)
    print(f"{region_id}: red={int((labels == 1).sum())}, blue={int((labels == 0).sum())}, saved={out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    for region_id in REGION_IDS:
        process_region(region_id, model, device)


if __name__ == "__main__":
    main()
