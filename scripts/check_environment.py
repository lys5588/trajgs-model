"""Check imports and a small CMA-VAE forward pass without external assets."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cuda", action="store_true", help="Also run a tiny CUDA tensor operation")
args = parser.parse_args()

import torch
import torchvision

versions = {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda}
kept = torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]),
                           torch.tensor([0.9, 0.8]), 0.5)
assert kept.tolist() == [0], kept
versions["torchvision_ops"] = "passed"
for name in ("numpy", "transformers", "accelerate", "deepspeed", "diffusers",
             "lightning", "pytorch_lightning", "spacy", "decord", "cv2"):
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "imported")

# Import the actual main-pipeline components, including local datasets/.
from models import Showo2WithMotion
from models.motion_vae import load_motion_vae
from datasets import create_motion_dataloader
from cma_vae.models.architectures.mld_vae_kl_align import CMAVAE

import cv2
import decord
import numpy as np

with tempfile.TemporaryDirectory(prefix="trajgs-video-") as temp_dir:
    video_path = Path(temp_dir) / "smoke.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    assert writer.isOpened(), "OpenCV video writer unavailable"
    for _ in range(3):
        writer.write(np.full((32, 32, 3), 80, dtype=np.uint8))
    writer.release()
    frames = decord.VideoReader(str(video_path)).get_batch([0, 1, 2]).asnumpy()
    assert frames.shape == (3, 32, 32, 3), frames.shape
versions["video_decode"] = "passed"

# Isolate HF datasets from the repository's same-named training package.
subprocess.run([sys.executable, "-I", "-c",
                "import datasets, evaluate, nlgmetricverse; print('NLG dependency imports passed')"],
               check=True)
versions["nlg_imports"] = "passed"

torch.set_num_threads(2)
torch.manual_seed(0)
vae = CMAVAE(ablation=SimpleNamespace(PE_TYPE="mld"), nfeats=269,
             latent_dim=[1, 32], ff_size=64, num_layers=1, num_heads=4,
             vision_dim=16, dropout=0.0).eval()
with torch.no_grad():
    latent, dist = vae.encode_motion(torch.randn(1, 16, 269), [16])
    reconstructed = vae.decode(dist.loc, [16])
assert reconstructed.shape == (1, 16, 269), reconstructed.shape
assert torch.isfinite(reconstructed).all()
versions["cma_vae_forward"] = "passed"

import spacy
nlp = spacy.load("en_core_web_sm")
assert len(nlp("A person walks forward.")) > 0
versions["spacy_model"] = "en_core_web_sm"

if args.cuda:
    assert torch.cuda.is_available(), "CUDA unavailable"
    with torch.no_grad():
        x = torch.ones((16, 16), device="cuda")
        assert (x @ x)[0, 0].item() == 16
    versions["cuda_smoke"] = "passed"
    versions["gpu"] = torch.cuda.get_device_name(0)
print(json.dumps(versions, indent=2))
