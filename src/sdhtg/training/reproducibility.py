from __future__ import annotations
import hashlib, json, os, platform, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool=True) -> None:
    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"]=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=not deterministic; torch.backends.cudnn.deterministic=deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def hash_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk:=stream.read(8*1024*1024): digest.update(chunk)
    return digest.hexdigest()


def hash_state_dict(model) -> str:
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_training_manifest(path: Path, *, seed: int, config_paths: list[str], data_paths: list[str], model) -> dict:
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit=None
    def safe_hash(file_path: str) -> str | None:
        if Path(file_path).is_file():
            return hash_file(file_path)
        return None
    value={"seed":seed,"git_commit":commit,"python":sys.version,"platform":platform.platform(),
           "torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
           "configs":{p:safe_hash(p) for p in config_paths},"data":{p:safe_hash(p) for p in data_paths},
           "initial_model_sha256":hash_state_dict(model)}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2),encoding="utf-8"); return value
