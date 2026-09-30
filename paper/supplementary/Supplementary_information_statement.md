# Text to paste into the manuscript

Replace the template placeholder under `\bmhead{Supplementary information}`
(currently: "If your article has accompanying supplementary file/s please state so here.")
with the following paragraph.

---

Supplementary information accompanies this paper. Supplementary Tables S1--S13
report: the paired statistical tests used for every comparative claim, including
Holm-corrected $q$ values and Cliff's $\delta$ (S1); the complete secondary
metrics of the main comparison, i.e.\ AUROC, precision, recall, MCC and balanced
accuracy for every dataset, method and seed (S2); seed-level results of the
imbalance-learning and prototype study (S3); the final hyper-parameter inventory
with the parameters that were and were not subjected to a sensitivity scan (S4);
the full hyper-parameter sensitivity scan (S5); the semantic- and entity-shortcut
ablations (S6); the template-level parsing-noise study at 20\% and 40\% rewrite
rates with paired clean references (S7); the entity-identifiability contrast under
the held-out-entity protocol (S8); the masked-template pretraining baseline
implemented under the controlled input protocol (S9); the reference values that
the cited methods report under their own protocols, together with protocol notes
(S10); the taxonomy of compared methods (S11); the negative-sampling pretraining
study (S12); and the provenance of the semantic fields together with the leakage
controls (S13). Supplementary Figures S1 and S2 show the training dynamics,
including a late-epoch validation collapse observed for one HDFS seed, and the
combined parameter-sensitivity and parsing-noise summary. Source data are provided
as a separate file and contain per-seed metrics for all 724 training runs, the
paired-test outputs and the efficiency benchmark records. The experiment protocol,
the run inventory and the SHA-256 hashes of the processed caches are provided as
Supplementary Text S1.

---

Note for the authors (delete before submission): the numbers quoted above are
exactly those contained in `tables/`, `figures/` and `source_data/` of this
folder; if you drop or rename any item, update this paragraph accordingly.
