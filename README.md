# Video Anomaly Detection with CLIP

This project focuses on video anomaly detection using a CLIP-based zero-shot learning approach.

Surveillance video frames are segmented and analyzed to detect abnormal events without additional model training.  
The goal is to explore zero-shot anomaly detection for real-world surveillance scenarios.

Instead of training a supervised classifier, this system compares visual embeddings with textual descriptions of anomalous events.

---

## Project Motivation

Traditional anomaly detection systems require labeled training data and task-specific models.  
In many real-world scenarios, collecting labeled anomaly data is difficult.

This project explores whether a pre-trained vision-language model (CLIP) can detect anomalies using only text descriptions.

---

## Method Overview

The system follows these steps:

1. Segment videos into fixed-length frame sequences.
2. Encode each frame using the CLIP image encoder.
3. Generate text embeddings for anomaly descriptions.
4. Compute similarity between visual and textual embeddings.
5. Apply threshold-based decision rules.
6. Aggregate segment-level predictions into video-level predictions.

This enables zero-shot and open-set anomaly detection.

---

## Project Structure
video-anomaly-clip/
│
├── src/ # Core inference pipeline
├── notebooks/ # Experiments and evaluation
├── outputs/ # Generated prediction CSV files
├── data/ # Manifest and dataset files
└── README.md


---

## Technologies

- Python
- PyTorch
- CLIP (Vision-Language Model)
- NumPy & Pandas
- Scikit-learn
- Jupyter Notebook

---

## Stage 2 – Zero-Shot Prototype

In this stage, we validated the feasibility of using CLIP for anomaly detection:

- Frame-level embedding extraction
- Segment-level similarity scoring
- Prompt-based anomaly detection
- Top-k voting strategy

Notebook:
notebooks/01_zero_shot_prototype.ipynb


---

## Stage 3 – Full Dataset-Level Inference

In this stage, we implemented:

- Dataset-level batch inference
- Segment-level anomaly scoring
- Video-level aggregation
- Balanced anomaly decision rule
- Confusion matrix and F1-score evaluation

Pipeline entry point:
python -m src.inference

Evaluation notebook:
notebooks/02_stage3_pipeline_evaluation.ipynb


---

## Threshold Tuning

We performed grid search over:

- Segment-level anomaly threshold
- Video-level anomaly ratio threshold

The objective was to maximize F1-score under a balanced anomaly decision rule.

Notebook:
notebooks/03_threshold_tuning.ipynb


---

## Evaluation Metrics

We evaluate performance using:

- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Status

Academic research project (TÜBİTAK 2209-A).

This project is under active development and will be extended with open-set anomaly detection and improved prompt engineering.