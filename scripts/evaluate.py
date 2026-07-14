from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch, yaml
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Sampler
from sdhtg.data.collate import collate_sessions, move_batch_to_device
from sdhtg.data.datasets import SessionDataset
from sdhtg.models.factory import build_model
from sdhtg.training.reproducibility import seed_everything

class BucketBatchSampler(Sampler):
    BUCKET_KEYS = [1,2,3,4,8,16,32,64,128,256,512]
    def __init__(self, lengths, batch_size, shuffle=False):
        buckets = {}
        for idx,l in enumerate(lengths):
            for b in self.BUCKET_KEYS:
                if l<=b: buckets.setdefault(b,[]).append(idx); break
        self.batches = []
        for _,indices in sorted(buckets.items()):
            key=_; bsz=256
            if key<=4: bsz=max(512,batch_size)
            elif key<=8: bsz=min(512,batch_size//4)
            elif key<=16: bsz=min(256,batch_size//8)
            elif key<=64: bsz=128
            else: bsz=64
            bsz=min(bsz,len(indices))
            for i in range(0,len(indices),bsz): self.batches.append(indices[i:i+bsz])
    def __len__(self): return len(self.batches)
    def __iter__(self): return iter(self.batches)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",required=True); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=256)
    a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(a.seed)
    cfg=yaml.safe_load(Path(a.config).read_text())
    processed=Path(cfg["data"]["processed_dir"])
    vocab=json.loads((processed/"vocab.json").read_text())
    ov={f"{n}_vocab_size":len(vocab[n]) for n in ["template","entity","action","status"]}
    model=build_model(cfg["model_config"],ov).to(device).eval()
    ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Checkpoint loaded  epoch={ckpt.get('epoch','?')}")
    ds=SessionDataset(str(processed/"sessions.parquet"),"test")
    sampler=BucketBatchSampler(ds.lengths,a.batch_size)
    loader=DataLoader(ds,batch_sampler=sampler,collate_fn=collate_sessions,
                      num_workers=int(cfg["data"]["num_workers"]),pin_memory=True)
    print(f"Test samples: {len(ds)}  batches: {len(loader)}")
    y_true,y_score=[],[]
    for batch in loader:
        batch=move_batch_to_device(batch,device)
        with torch.inference_mode():
            out=model(batch)
        y_true.extend(batch["label"].cpu().tolist())
        y_score.extend(out.anomaly_probability.float().cpu().tolist())
    y_true=np.asarray(y_true,dtype=int); y_score=np.asarray(y_score,dtype=float)
    auprc=float(average_precision_score(y_true,y_score))
    auroc=float(roc_auc_score(y_true,y_score)) if len(np.unique(y_true))>1 else 0.0
    print(f"\nAUPRC: {auprc:.4f}   AUROC: {auroc:.4f}")
    ths=np.unique(np.quantile(y_score,np.linspace(0,1,2001)))
    bf,bth,bp,br=0,0,0,0
    for th in ths:
        yp=(y_score>=th).astype(int); f1=f1_score(y_true,yp,zero_division=0)
        if f1>bf: bf,bth,bp,br=f1,th,precision_score(y_true,yp,zero_division=0),recall_score(y_true,yp,zero_division=0)
    print(f"Best F1 threshold: {bth:.4f}")
    print(f"  Precision: {bp:.4f}   Recall: {br:.4f}   F1: {bf:.4f}")
    res={"auprc":auprc,"auroc":auroc,"threshold":bth,"precision":bp,"recall":br,"f1":bf,"samples":int(len(y_true))}
    out_path=Path(a.checkpoint).parent/"test_result.json"
    out_path.write_text(json.dumps(res,indent=2),encoding="utf-8")
    print(f"Saved to {out_path}")

if __name__=="__main__": main()
