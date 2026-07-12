from __future__ import annotations
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sdhtg.data.collate import move_batch_to_device
from sdhtg.evaluation.calibration import calibrate_threshold
from sdhtg.losses.contrastive import ProjectionHead, supervised_info_nce
from .curriculum import Curriculum
from .checkpoint import CheckpointManager
from .reproducibility import hash_state_dict


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, validation_loader, config, output_dir, device):
        self.model=model; self.criterion=criterion; self.optimizer=optimizer
        self.train_loader=train_loader; self.validation_loader=validation_loader; self.config=config
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True); self.device=torch.device(device)
        self.amp=bool(config.get("amp",True) and self.device.type=="cuda")
        self.scaler=torch.amp.GradScaler("cuda",enabled=self.amp); self.curriculum=Curriculum(config["curriculum"])
        self.checkpoints=CheckpointManager(self.output_dir/"checkpoints"); self.global_step=0
        self.best_metric=-float("inf") if config["monitor_mode"]=="max" else float("inf"); self.patience_count=0

    def _autocast(self): return torch.autocast(device_type="cuda",dtype=torch.float16) if self.amp else nullcontext()

    def train_epoch(self, epoch):
        self.model.train(); state=self.curriculum.at(epoch,self.config["max_epochs"]); totals=[]
        self.optimizer.zero_grad(set_to_none=True); accumulation=int(self.config["grad_accumulation_steps"])
        for index,batch in enumerate(self.train_loader):
            batch=move_batch_to_device(batch,self.device)
            mask_prob=self.config.get("mask_template_prob",0.0)
            if mask_prob>0:
                tm=(torch.rand_like(batch["delta_t"])<mask_prob)&batch["mask"]
                batch["template_id"]=batch["template_id"].masked_fill(tm,1)
            with self._autocast():
                output=self.model(batch,boundary_temperature=state.boundary_temperature,film_strength=state.film_strength)
                loss=self.criterion(output,batch["label"],boundary_scale=state.boundary_loss_scale).total/accumulation
            self.scaler.scale(loss).backward()
            if (index+1)%accumulation==0 or index+1==len(self.train_loader):
                self.scaler.unscale_(self.optimizer); nn.utils.clip_grad_norm_(self.model.parameters(),self.config["grad_clip_norm"])
                self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True); self.global_step+=1
            totals.append(float(loss.detach())*accumulation)
        return {"loss":float(np.mean(totals)),"temperature":state.boundary_temperature,"film_strength":state.film_strength}

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval(); labels=[]; scores=[]; losses=[]
        for batch in loader:
            batch=move_batch_to_device(batch,self.device)
            with self._autocast():
                output=self.model(batch); result=self.criterion(output,batch["label"])
            labels.extend(batch["label"].cpu().tolist()); scores.extend(output.anomaly_probability.float().cpu().tolist()); losses.append(float(result.total))
        return np.asarray(labels),np.asarray(scores),float(np.mean(losses))

    def fit(self, resume=None):
        start=0
        if resume:
            payload=self.checkpoints.load(resume,model=self.model,optimizer=self.optimizer,scaler=self.scaler,map_location=self.device)
            start=payload["epoch"]+1; self.global_step=payload["global_step"]; self.best_metric=payload["best_metric"]; self.patience_count=payload["patience_count"]
        history=[]
        for epoch in range(start,self.config["max_epochs"]):
            train=self.train_epoch(epoch); y,s,val_loss=self.predict(self.validation_loader)
            metrics={"auprc":float(average_precision_score(y,s)),"auroc":float(roc_auc_score(y,s)) if len(np.unique(y))>1 else float("nan"),"loss":val_loss}
            monitored=metrics[self.config["monitor"]]; improved=(monitored>self.best_metric+self.config["minimum_delta"] if self.config["monitor_mode"]=="max" else monitored<self.best_metric-self.config["minimum_delta"])
            if improved: self.best_metric=monitored; self.patience_count=0
            else: self.patience_count+=1
            record={"epoch":epoch,"train":train,"validation":metrics}; history.append(record)
            metadata={"record":record,"model_sha256":hash_state_dict(self.model)}
            self.checkpoints.save("last",model=self.model,optimizer=self.optimizer,scheduler=None,scaler=self.scaler,epoch=epoch,global_step=self.global_step,best_metric=self.best_metric,patience_count=self.patience_count,metadata=metadata)
            if improved: self.checkpoints.save("best",model=self.model,optimizer=self.optimizer,scheduler=None,scaler=self.scaler,epoch=epoch,global_step=self.global_step,best_metric=self.best_metric,patience_count=0,metadata=metadata)
            (self.output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            if self.patience_count>=self.config["patience"]: break
        self.checkpoints.load(self.output_dir/"checkpoints/best.pt",model=self.model,map_location=self.device)
        y,s,_=self.predict(self.validation_loader); threshold=calibrate_threshold(y,s,**self.config["threshold"])
        (self.output_dir/"threshold.json").write_text(json.dumps(threshold,indent=2),encoding="utf-8")
        return {"best_metric":self.best_metric,"threshold":threshold,"epochs":len(history)}
