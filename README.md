ProactiveGuard: Multi-Modal Deep Learning for Real-Time Threat Detection
A multi‑modal deep learning framework for proactive threat detection in intelligent video surveillance.
It unifies weapon detection (YOLOv10s), violence recognition (MobileNetV2 + Bi‑LSTM + Self‑Attention),
fear‑emotion analysis (Vision Transformer), and behavioural precursor scoring into a single, interpretable
threat score.

Table of Contents

Overview

Repository Structure

System Architecture

Results Summary

Setup & Installation

Dataset Preparation

Training the Modules

1. Violence Detection

2. Weapon Detection

3. Emotion (Fear) Detection

Fusion Training & Full Evaluation

Ethical Considerations

Citation

License

Overview

Modern CCTV systems are largely reactive – they record incidents but cannot anticipate them.
ProactiveGuard addresses this gap by combining four complementary detection streams into a
unified Proactive Threat Score.

Weapon Detection: YOLOv10s trained on a merged multi‑source dataset (single “weapon” class).

Violence Recognition: MobileNetV2 + Bidirectional LSTM with temporal self‑attention.

Emotion Analysis: Vision Transformer (ViT‑Base/16) fine‑tuned on AffectNet+RAF‑DB+FER2013 for fear detection.

Behavioural Precursor: Optical‑flow‑based motion variance heuristic that captures loitering, nervous movement, and concealment gestures.

A learnable MLP fusion head combines the four scores into a continuous threat score between 
(0 – 1) that is thresholded into Low / Medium / High alerts.
The system achieves 96.5% accuracy and 0.985 AUC‑ROC on a balanced test set,
and demonstrates proactive detection up to 33 frames (~1.1 s) before incident escalation.

Repository Structure


├── proactiveguard‑violence‑detector.ipynb

    # Violence dataset preparation & training
    
├── proactiveguard‑weapons‑detector.ipynb

    # Weapons dataset merging & YOLOv10s training
    
├── ProactiveGuard emotion.ipynb

    # Emotion dataset merging & ViT training
    
├── fusion‑and‑evaluation‑notebook.ipynb

    # Fusion head training, full evaluation suite, ablation, lead‑time analysis
    
└── README.md

All notebooks are designed to run on Kaggle (NVIDIA Tesla T4 GPUs) but can be adapted to any environment with a GPU and sufficient RAM.

System Architecture


Four parallel modules process every incoming video frame/clip:


Weapon Module

Model: YOLOv10s (NMS‑free, COCO‑pretrained)

Output: W(t) – max detection confidence in frame *t*


Violence Module

Spatial encoder: MobileNetV2 (frozen layers 0‑9)

Temporal encoder: 2‑layer Bi‑LSTM (hidden=256) + temporal self‑attention

Output: V(t) – posterior probability of “violent” class over a 32‑frame window


Emotion Module

Model: ViT‑Base/16 (86M params) with custom classification head

Output: F(t) – fear class probability, averaged over 8 middle frames of a clip


Precursor Module

Farnebäck dense optical flow between sampled frames

Output: P(t) – coefficient of variation of flow magnitude, normalised to [0,1]


Fusion Head (MLP: 4 → 16 → 8 → 1)
Combines the four scores into a threat logit, then sigmoid to obtain
S_threat(t). Alert thresholds:

Threat Level | Condition | Action
Low	S_threat < 0.40	Log only
Medium	0.40 ≤ S < 0.50	Dashboard alert to operator
High	S_threat ≥ 0.50	Immediate notification


Results Summary


Module Performance (on held‑out test sets)

Module | Accuracy |	F1 | AUC | mAP@0.5

Violence (PG)	96.5%	96.5%	98.3%	–

Emotion – Fear (PG)	89.7%	67.1%	92.1%	–

Weapon (PG)	–	–	–	93.1%

Fusion System (ProactiveGuard)

Metric | Value

Accuracy	0.9650

Precision	0.9604

Recall	0.9700

F1‑score	0.9652

FPR	0.0400

FNR	0.0300

AUC‑ROC	0.9846

PR‑AUC	0.9838

Ablation Study (F1 / AUC‑ROC)

Configuration | F1 (%) | AUC‑ROC

Full system	96.5	0.9846

No Violence	0.0	0.7584

No Weapon	96.5	0.9845

No Precursor	96.5	0.9791

No Emotion	96.5	0.9809

Violence only	96.5	0.9833

Weapon only	0.0	0.4845

Violence is the dominant signal, but all modalities contribute to robustness and lower false positive rates.

Proactive Lead‑Time

Detection rate: 100% (2000 simulations)

Mean lead time: 33.4 frames (~1.11 s at 30 FPS)

Std: 4.4 frames

See the paper for full details.

Setup & Installation:

Requirements

Python 3.8+

CUDA‑compatible GPU (recommended; the models can run on CPU but will be slow)

Kaggle API credentials (optional, for automatic dataset download)

Install Dependencies:

bash
pip install ultralytics torch torchvision torchaudio tqdm scikit-learn opencv-python pandas matplotlib seaborn pyyaml kagglehub transformers datasets accelerate

Kaggle API Setup (for data download)

If you intend to use the automatic download functions inside the notebooks, set your Kaggle credentials:

bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

Or place kaggle.json in ~/.kaggle/. Alternatively, you can manually download the datasets from Kaggle and adjust the paths in the notebooks.

Dataset Preparation

All datasets are publicly available. The notebooks contain code to download and pre‑process them, but you may also manually place them in the expected directories.

Violence Dataset:

Source: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

Kaggle slug: mohamedmustafa/real-life-violence-situations-dataset

The notebook extracts 32‑frame sequences from each video and organises them into:

text
data/violence/frames/{train,val,test}/{violence,nonviolence}/{clip_id}/frame_xxxx.jpg


Weapon Dataset:

Merged from four public sources into a single “weapon” class:

Weapons detection dataset, available at:
https://www.kaggle.com/datasets/snehilsanyal/weapon-detection-test

weapon detection cctv v3 dataset Computer Vision Model, available at:
https://universe.roboflow.com/weapon-detection-cctv/weapon-detection-cctv-v3-dataset

weapondetection2fps Computer Vision Dataset, available at:
https://universe.roboflow.com/scan-gizi-makanan/weapondetection2fps

Final Computer Vision Dataset, available at:
https://universe.roboflow.com/detection-gun/final-hcrmc-tsvcy

The merging notebook (proactiveguard‑weapons‑detector.ipynb) handles class re‑mapping, train/val/test splitting, and generation of a unified data.yaml.
After merging, the dataset is stored at /kaggle/working/unified_dataset_one_class/.

Emotion Dataset:

Merged from three sources:

FER2013 (Kaggle pre‑processed version) - https://www.kaggle.com/datasets/msambare/fer2013

AffectNet (archive) - https://www.kaggle.com/datasets/mstjebashazida/affectnet

RAF‑DB - https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

All three are mapped to 7 classes: angry, disgust, fear, happy, sad, surprise, neutral.
A stratified 70/15/15 split is performed per class, resulting in ~53k training images.
The symlinked merged dataset is created at /kaggle/working/emotion_combined/.

Training the Modules
Each module can be trained independently. The notebooks are self‑contained and include all steps from data preparation to checkpoint saving.

1. Violence Detection

Notebook: proactiveguard‑violence‑detector.ipynb

Prepares the violence frame dataset.

Defines ViolenceDetector (MobileNetV2 + Bi‑LSTM + Self‑Attention).

Trains with AdamW, class‑weighted sampling, early stopping.

Saves best model to checkpoints/violence_best.pt.

Key Hyperparameters (config in notebook):

seq_len=32, lstm_hidden=256, lstm_layers=2, dropout=0.4

epochs: 60, batch: 8, lr: 3e‑4, patience: 12

Output: violence_best.pt (model state dict + validation metrics).

2. Weapon Detection

Notebook: proactiveguard‑weapons‑detector.ipynb

Merges four weapon datasets into one binary class (“weapon”).

Trains YOLOv10s using ultralytics (80‑100 epochs, batch 16, imgsz 640).

Saves best weights to /kaggle/working/weapon_best.pt.

Key settings:

Single‑class mode (single_cls=True)

Augmentations: mosaic, mixup, copy_paste, flips, etc.

Early stopping patience: 20

Output: /kaggle/working/weapon_best.pt

3. Emotion (Fear) Detection

Notebook: ProactiveGuard emotion.ipynb

Merges AffectNet, RAF‑DB, FER2013 into emotion_combined/.

Defines EmotionViT (ViT‑Base/16 with custom head).

Trains with warm‑up, cosine annealing, early stopping (patience 6).

Saves best model based on fear class F1 to /kaggle/working/emotion_vit_best.pt.

Key Hyperparameters:

batch_size=32, epochs=17, lr=1e‑4, warmup_epochs=3

Transforms: RandomResizedCrop, HorizontalFlip, ColorJitter

Output: emotion_vit_best.pt (full model state dict).

Fusion Training & Full Evaluation

Notebook: fusion‑and‑evaluation‑notebook.ipynb

This notebook performs:

Pre‑computation of module scores
The frozen violence, weapon, and emotion models generate confidence scores for every clip in the violence frames directory. The precursor score is computed on the fly. These are saved to fusion_scores.csv.

Fusion Head Training
An MLP (4→16→8→1) is trained to combine the four scores into a binary threat prediction.

Can operate in fixed‑weight mode (using predefined priority) or learnable mode.

Best checkpoint saved as checkpoints/fusion_best.pt.

Full Evaluation Suite

Module‑level metrics (violence, emotion, weapon mAP).

Fusion system metrics (accuracy, F1, AUC‑ROC, confusion matrix).

Ablation study (zeroing out each module).

Proactive lead‑time simulation.

Generates plots (evaluation report) and a JSON results file.

Run this notebook after the three specialist modules have been trained and their checkpoints are available. Adjust the checkpoint paths in the notebook’s config to point to your trained .pt files.

Ethical Considerations:

ProactiveGuard is designed as an assistive tool – it does not autonomously trigger enforcement actions.
Key safeguards mentioned in the paper:

Event‑driven processing: only motion‑triggered frames should be transmitted, not continuous video.

Face anonymisation: faces of low‑threat individuals should be blurred before storage.

Human‑in‑the‑loop: all alerts are reviewed by a human operator before any action.

Demographic fairness: current training data is predominantly Western‑centric; auditing for bias across ethnic groups is an essential future step.

Deployment should comply with local privacy regulations (e.g., GDPR) and include transparent policies on data retention and algorithmic decision‑making.

Citation
If you use this work in your research, please cite the accompanying paper:

Victor, M. U., & Stephen, B. U. (2026). ProactiveGuard: A Multi‑Modal Deep Learning Framework for Real‑Time Threat Detection Integrating YOLOv10, CNN‑BiLSTM, and Vision Transformer Fusion in Intelligent Surveillance. Proceedings of ICCOMTECH.

BibTeX:

bibtex
@inproceedings{victor2026proactiveguard,
  title={ProactiveGuard: A Multi-Modal Deep Learning Framework for Real-Time Threat Detection Integrating YOLOv10, CNN-BiLSTM, and Vision Transformer Fusion in Intelligent Surveillance},
  author={Victor, Mfoniso Utibeabasi and Stephen, Bliss Utibe-Abasi},
  booktitle={International Conference on Communication Technology (ICCOMTECH)},
  year={2026}
}
License
The code is provided for research and educational purposes. Please consult the individual dataset licences before commercial use. The authors encourage open collaboration and welcome pull requests.



