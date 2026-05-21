# Video Anomaly Detection with CLIP

A zero-shot video anomaly detection system built on OpenAI's CLIP (Contrastive Language-Image Pretraining). The system classifies surveillance video segments as normal or anomalous without any task-specific fine-tuning, leveraging CLIP's vision-language alignment through prompt-based similarity scoring.

**Academic context:** TÜBİTAK 2209-A Undergraduate Research Support Programme.

---

## Approach

Each video is divided into fixed-length segments. For every segment, frame-level CLIP similarity scores are computed against a set of natural language prompts describing anomalous and normal events. Segment-level scores are aggregated to a video-level decision using a top-K mean strategy, and a threshold calibrated on the validation set determines the final binary prediction.

The pipeline requires no labeled training data and no gradient updates to the model — classification emerges entirely from the semantic structure of CLIP's embedding space.

---

## Architecture

```
Video
 └── Segments (fixed-length)
      └── Frames
           └── CLIP Encoder
                ├── Frame × Anomaly Prompts  →  anomaly_sim[]
                └── Frame × Normal Prompts   →  normal_sim[]
                         ↓
              Per-segment score aggregation
                         ↓
              Top-K mean over segments  →  score_diff
                         ↓
              Threshold comparison  →  is_anomaly (bool)
```

**Scoring:** `score_diff = topk_mean(anomaly_sim) − topk_mean(normal_sim)`

A positive `score_diff` indicates anomalous content. The threshold and top-K ratio are jointly calibrated on the validation set by maximizing balanced accuracy.

---

## Project Structure

```
video-anomaly-clip/
├── src/
│   ├── config.py          # Prompt definitions (ANOMALY_CLASSES, NORMAL_CLASSES)
│   ├── inference.py       # CLIPInference — frame-level similarity scoring
│   ├── aggregation.py     # aggregate_video_scores — segment → video aggregation
│   ├── metrics.py         # Evaluation utilities
│   └── pipeline.py        # End-to-end real-time inference pipeline
├── notebooks/
│   └── 02_stage3_pipeline_evaluation_val_test.ipynb  # Full evaluation notebook
├── data/
│   ├── manifests_new/     # manifest.csv (segment metadata, splits)
│   └── segments_new/      # Preprocessed video segments (val / test)
├── artifacts/
│   └── threshold.json     # Calibrated threshold and top-K ratio
├── outputs/               # Predictions, plots, confusion matrices
└── requirements.txt
```

---

## Dataset

Experiments use a subset of the [UCF-Crime dataset](https://www.crcv.ucf.edu/projects/real-world/), covering 12 anomaly categories and normal video footage.

| Split | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| Val   | 417    | 661     | 1 078 segments |
| Test  | 414    | 688     | 1 102 segments |

Anomaly categories: Abuse, Arrest, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism.

---

## Results

Threshold and top-K ratio calibrated on the validation set (`TOPK_RATIO=0.30`, `threshold=−0.000795`).

### Validation Set

| Metric | Score |
|--------|-------|
| Balanced Accuracy | 0.8139 |
| AUC-ROC | 0.8208 |
| Recall (Sensitivity) | 0.778 |
| Specificity | 0.850 |

### Test Set

| Metric | Score |
|--------|-------|
| Balanced Accuracy | 0.6889 |
| AUC-ROC | 0.7431 |

The val→test gap reflects the inherent difficulty of zero-shot generalization across unseen video instances with no domain adaptation.

---

## Key Design Decisions

- **Top-K aggregation** over mean/max: reduces sensitivity to outlier frames while preserving the strongest anomaly signal within a segment.
- **Balanced accuracy** as calibration objective: handles class imbalance in the UCF-Crime distribution without resampling.
- **Zero-shot prompts**: class-specific natural language descriptions rather than generic anomaly/normal labels, exploiting CLIP's fine-grained semantic understanding.

---

## Dependencies

See `requirements.txt`. Core dependencies: `torch`, `torchvision`, `openai/CLIP`, `scikit-learn`, `opencv-python`.