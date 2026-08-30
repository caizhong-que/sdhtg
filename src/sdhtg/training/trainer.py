from __future__ import annotations
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from sdhtg.data.collate import move_batch_to_device
from sdhtg.data.shortcuts import (
    mask_entity_to_unk,
    mask_explicit_status_words,
    shuffle_entity_ids,
)
from sdhtg.evaluation.calibration import calibrate_threshold
from sdhtg.losses.contrastive import ProjectionHead, supervised_info_nce
from .curriculum import Curriculum
from .checkpoint import CheckpointManager
from .reproducibility import hash_state_dict


logger = logging.getLogger("sdhtg.trainer")


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, validation_loader, config, output_dir, device,
                 test_loader=None, entity_vocab_size=None, status_word_mask=None):
        self.model=model; self.criterion=criterion; self.optimizer=optimizer
        self.train_loader=train_loader; self.validation_loader=validation_loader; self.config=config
        self.test_loader=test_loader
        self.entity_vocab_size=entity_vocab_size
        self.status_word_mask=status_word_mask
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True); self.device=torch.device(device)
        self.amp=bool(config.get("amp",True) and self.device.type=="cuda")
        self.scaler=torch.amp.GradScaler("cuda",enabled=self.amp); self.curriculum=Curriculum(config["curriculum"])
        self.checkpoints=CheckpointManager(self.output_dir/"checkpoints"); self.global_step=0
        self.best_metric=-float("inf") if config["monitor_mode"]=="max" else float("inf"); self.patience_count=0

    def _autocast(self): return torch.autocast(device_type="cuda",dtype=torch.float16) if self.amp else nullcontext()

    def _apply_shortcuts(self, batch, seed=0):
        if self.config.get("entity_to_unk"):
            mask_entity_to_unk(batch)
        if self.config.get("entity_id_shuffle") and self.entity_vocab_size:
            shuffle_entity_ids(batch, self.entity_vocab_size, seed)
        if self.config.get("mask_explicit_status_words") and self.status_word_mask:
            mask_explicit_status_words(batch, self.status_word_mask)
        return batch

    def train_epoch(self, epoch):
        st = self.curriculum.at(epoch,self.config["max_epochs"])
        logger.info("epoch %d start  temp=%.2f  film=%.2f", epoch, st.boundary_temperature, st.film_strength)
        self.model.train()
        state=self.curriculum.at(epoch,self.config["max_epochs"])
        totals=[]
        self.optimizer.zero_grad(set_to_none=True)
        accumulation=int(self.config["grad_accumulation_steps"])
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", unit="batch")
        for index,batch in enumerate(pbar):
            batch=move_batch_to_device(batch,self.device)
            batch=self._apply_shortcuts(batch, seed=self.global_step)
            mask_prob=self.config.get("mask_template_prob",0.0)
            if mask_prob>0:
                tm=(torch.rand_like(batch["delta_t"])<mask_prob)&batch["mask"]
                batch["template_id"]=batch["template_id"].masked_fill(tm,1)
            with self._autocast():
                output=self.model(batch,boundary_temperature=state.boundary_temperature,film_strength=state.film_strength)
                loss=self.criterion(
                    output, batch["label"],
                    boundary_scale=state.boundary_loss_scale,
                    action_change=batch["action_change"],
                    entity_change=batch["entity_change"],
                    template_id=batch["template_id"],
                    boundary_label=batch.get("boundary_label"),
                ).total/accumulation
            self.scaler.scale(loss).backward()
            if (index+1)%accumulation==0 or index+1==len(self.train_loader):
                # Gradient accumulation boundary: unscale once, clip, step.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),self.config["grad_clip_norm"])
                self.scaler.step(self.optimizer); self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            self.global_step+=1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            totals.append(float(loss.detach())*accumulation)
        logger.info("epoch %d train  loss=%.4f  batches=%d", epoch, float(np.mean(totals)), len(self.train_loader))
        return {"loss":float(np.mean(totals)),"temperature":state.boundary_temperature,"film_strength":state.film_strength}

    @torch.inference_mode()
    def predict(self, loader):
        self.model.eval(); labels=[]; scores=[]; losses=[]
        for batch in loader:
            batch=move_batch_to_device(batch,self.device)
            batch=self._apply_shortcuts(batch, seed=0)
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
            logger.info("epoch %d  trn=%.4f  val=%.4f  auprc=%.4f  auroc=%.4f%s",
                        epoch, train["loss"], val_loss, metrics["auprc"], metrics["auroc"],
                        " *" if improved else "")
            record={"epoch":epoch,"train":train,"validation":metrics}; history.append(record)
            metadata={"record":record,"model_sha256":hash_state_dict(self.model)}
            self.checkpoints.save("last",model=self.model,optimizer=self.optimizer,scheduler=None,scaler=self.scaler,epoch=epoch,global_step=self.global_step,best_metric=self.best_metric,patience_count=self.patience_count,metadata=metadata)
            if improved: self.checkpoints.save("best",model=self.model,optimizer=self.optimizer,scheduler=None,scaler=self.scaler,epoch=epoch,global_step=self.global_step,best_metric=self.best_metric,patience_count=0,metadata=metadata)
            (self.output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            if self.patience_count>=self.config["patience"]:
                logger.info("early stopping  epoch=%d  best=%.4f", epoch, self.best_metric); break
        self.checkpoints.load(self.output_dir/"checkpoints/best.pt",model=self.model,map_location=self.device)
        y,s,_=self.predict(self.validation_loader); threshold=calibrate_threshold(y,s,**self.config["threshold"])
        logger.info("calibrated  threshold=%.4f  f1=%.4f", threshold["threshold"], threshold.get("validation_value",0))
        (self.output_dir/"threshold.json").write_text(json.dumps(threshold,indent=2),encoding="utf-8")
        # --- test set evaluation ---
        test_result = None
        if self.test_loader is not None:
            logger.info("evaluating test set")
            y_test, s_test, test_loss = self.predict(self.test_loader)
            if len(np.unique(y_test)) > 1:
                test_auprc = float(average_precision_score(y_test, s_test))
                test_auroc = float(roc_auc_score(y_test, s_test))
            else:
                test_auprc = float(average_precision_score(y_test, s_test))
                test_auroc = float("nan")
            th = threshold["threshold"]
            yp = (s_test >= th).astype(int)
            test_precision = float(precision_score(y_test, yp, zero_division=0))
            test_recall = float(recall_score(y_test, yp, zero_division=0))
            test_f1 = float(f1_score(y_test, yp, zero_division=0))
            test_result = {
                "auprc": test_auprc,
                "auroc": test_auroc,
                "threshold_applied": th,
                "precision": test_precision,
                "recall": test_recall,
                "f1": test_f1,
                "loss": test_loss,
                "samples": int(len(y_test)),
            }
            logger.info("test  auprc=%.4f  auroc=%.4f  f1=%.4f  p=%.4f  r=%.4f",
                        test_auprc, test_auroc, test_f1, test_precision, test_recall)
            test_result_path = self.output_dir / "test_result.json"
            if test_result_path.exists():
                existing = json.loads(test_result_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = [existing]
                existing.append(test_result)
                test_result_path.write_text(
                    json.dumps(existing, indent=2), encoding="utf-8"
                )
            else:
                test_result_path.write_text(
                    json.dumps([test_result], indent=2), encoding="utf-8"
                )
            # Append to CSV for easy viewing
            csv_path = self.output_dir / "test_results.csv"
            csv_fields = ["auprc","auroc","threshold_applied","precision","recall","f1","loss","samples"]
            file_exists = csv_path.exists()
            with csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(test_result)
        return {"best_metric":self.best_metric,"threshold":threshold,"epochs":len(history),"test":test_result}
