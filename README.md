SkinBench-v2: Multilayer Skin Disease Classification (ALL-9 + L1/L2 + Subclass L3)

TL;DR
A production-ready, reproducible pipeline for skin-disease image classification with a multilayer router:
L1 (Normal vs Abnormal) → L2 (8 abnormal classes) → optional L3 (Subclass) under Eczema / Fungal / Pox.
Models compared: ResNet50, DenseNet121, MobileNetV3, CNN baseline, ViT-ResNet hybrid, Swin-DenseNet hybrid, EfficientNet, VGG16, VGG19.

1) Project Structure (as used in this repo)
SkinBench/
├─ data_raw/
│  ├─ train/  ├─ val/  └─ test/
│  │   ├─ Acne/  ├─ Bacterial Infections/  ├─ Eczema/
│  │   ├─ Fungal Infections/  ├─ Normal/  ├─ Pigmentation Disorders/
│  │   ├─ Pox/  ├─ Psoriasis/  └─ Scabies/
├─ models/                  # model definitions (cnn, resnet, densenet, vgg, efficientnet, vit_resnet, swin_densenet, etc.)
├─ eval_tools/              # evaluation scripts (run_all_evals, ROC/PR, reliability, CM, McNemar, etc.)
├─ runs/
│  ├─ L1/                   # L1 (Normal vs Abnormal) checkpoints & outputs
│  │   ├─ resnet50/best.pt
│  │   └─ ...
│  ├─ L2/                   # L2 (8-class abnormal) checkpoints & outputs
│  │   ├─ resnet50/best.pt
│  │   ├─ densenet121/best.pt
│  │   └─ ...
│  ├─ ALL9/                 # single-stage 9-class experiments
│  │   ├─ cnn/best.pt
│  │   ├─ mobilenetv3/best.pt
│  │   ├─ vit_resnet/best.pt
│  │   ├─ swin_densenet/best.pt
│  │   ├─ efficientnet/best.pt
│  │   ├─ vgg16/best.pt
│  │   ├─ vgg19/best.pt
│  │   └─ ...
│  ├─ SUBCLASS/
│  │   └─ subclass/best.pt  # optional L3 subclass head (Eczema/Fungal/Pox subtypes)
│  ├─ figures/              # all plots auto-saved here
│  └─ tables/               # all CSV outputs here (predictions, confusion, leaderboard, comparison)
├─ train_multilayer.py
├─ datasets.py
├─ app.py                   # Streamlit app (L1 + L2 + optional L3)
└─ README.md

2) Data Assumptions

Directory name must be data_raw with train/, val/, test/ splits.

ALL-9 classes (folder names):
Acne, Bacterial Infections, Eczema, Fungal Infections, Normal, Pigmentation Disorders, Pox, Psoriasis, Scabies

3) Multilayer Method

L1 (binary): Normal vs Abnormal

Trained with --phase L1 (e.g., ResNet50 best)

L2 (8 classes): for Abnormal images only

Classes: Acne, Bacterial Infections, Eczema, Fungal Infections, Pigmentation Disorders, Pox, Psoriasis, Scabies

Trained with --phase L2 (e.g., DenseNet121 / ResNet50 best)

Optional L3 (Subclass): when L2 predicts Eczema or Fungal Infections or Pox, an extra head refines sub-classes.

Trained with --phase SUBCLASS and your subclass labels inside datasets.py (see comments in the file).

Inference path in app.py: L1 → L2 → if {Eczema/Fungal/Pox} → L3.

4) Environment
conda create -n torch_gpu python=3.10 -y

conda activate torch_gpu

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install timm==0.9.16 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2 pandas==2.2.2

pip install streamlit==1.37.0

6) Training
L1 (Normal vs Abnormal)

python train_multilayer.py --data_dir data_raw --phase L1 --model resnet50

# checkpoints → runs/L1/resnet50/best.pt

L2 (8-class abnormal)
python train_multilayer.py --data_dir data_raw --phase L2 --model densenet121

# checkpoints → runs/L2/densenet121/best.pt

ALL-9 (single-stage baseline & comparisons)
# pick any:
python train_multilayer.py --data_dir data_raw --phase ALL9 --model mobilenetv3

python train_multilayer.py --data_dir data_raw --phase ALL9 --model vit_resnet

python train_multilayer.py --data_dir data_raw --phase ALL9 --model swin_densenet

python train_multilayer.py --data_dir data_raw --phase ALL9 --model efficientnet

python train_multilayer.py --data_dir data_raw --phase ALL9 --model vgg16

python train_multilayer.py --data_dir data_raw --phase ALL9 --model vgg19

python train_multilayer.py --data_dir data_raw --phase ALL9 --model cnn


L3 (Subclass head) — optional
python train_multilayer.py --data_dir data_raw --phase SUBCLASS --model densenet121

# checkpoint → runs/SUBCLASS/subclass/best.pt
# NOTE: align subclass labels in datasets.py (Eczema/Fungal/Pox subtypes).

6) Evaluation (auto-plots + CSVs)

Run all evaluations and comparisons in one go:

python -m eval_tools.run_all_evals --data_dir data_raw --phases ALL9 L1 L2 --models resnet50 densenet121 mobilenetv3 cnn vit_resnet swin_densenet efficientnet vgg16 vgg19
# outputs:
#   runs/figures/*.png     (confusion, ROC-OVR, PR-OVR, reliability)
#   runs/tables/*.csv      (predictions, confusion, leaderboard, comparison_all_models.csv)



7) Streamlit Inference (with subclass)

app.py loads:

L1: runs/L1/resnet50/best.pt

L2: runs/L2/densenet121/best.pt

L3 (optional): runs/SUBCLASS/subclass/best.pt

If your L3 checkpoint was trained with a slightly different architecture, set:

STRICT_L3_LOAD = False


inside app.py (already handled) to bypass minor key mismatches during weight load.

Run:

streamlit run app.py

8) Results: Quick Leaderboard

Full table: runs/tables/comparison_all_models.csv

(Examples from your last eval run)

Phase	Model	Acc	Macro-F1
ALL9	ViT-ResNet	0.9843	0.9836
ALL9	Swin-DenseNet	0.9843	0.9839
ALL9	EfficientNet	0.9833	0.9827
ALL9	MobileNetV3	0.9815	0.9808
ALL9	VGG16	0.9721	0.9713
ALL9	VGG19	0.9667	0.9668
ALL9	CNN (baseline)	0.6389	0.6220
L1	ResNet50	0.9980	0.9959
L2	DenseNet121	0.9853	0.9849
L2	ResNet50	0.9827	0.9824

L1+L2 cascade is used for the Streamlit deploy; L3 is triggered for Eczema/Fungal/Pox to predict subclasses.

9) Plots (render automatically on GitHub)

All image links use relative paths (no hardcoded localhost). After you push to v2, GitHub will display them inline.

9.1 Confusion Matrices (ALL-9)

CNN:

EfficientNet:

MobileNetV3:

Swin-DenseNet:

ViT-ResNet:

VGG16:

VGG19:

9.2 Confusion Matrices (L1 / L2)

L1-ResNet50:

L2-ResNet50:

L2-DenseNet121:

9.3 ROC-OVR (ALL-9)

CNN:

EfficientNet:

MobileNetV3:

Swin-DenseNet:

ViT-ResNet:

VGG16:

VGG19:

9.4 PR-OVR (ALL-9)

CNN:

EfficientNet:

MobileNetV3:

Swin-DenseNet:

ViT-ResNet:

VGG16:

VGG19:

9.5 Reliability (ALL-9)

CNN:

EfficientNet:

MobileNetV3:

Swin-DenseNet:

ViT-ResNet:

VGG16:

VGG19:

9.6 Binary (L1) — ROC/PR/Reliability

ROC:

PR:

Reliability:

9.7 L2 (8-class) — ROC/PR/Reliability

ResNet50 ROC:

ResNet50 PR:

ResNet50 Reliability:

DenseNet121 ROC:

DenseNet121 PR:

DenseNet121 Reliability:

10) CSV Outputs (auto-generated)

Master comparison: runs/tables/comparison_all_models.csv

Per-model predictions (ALL-9):

pred_ALL9_cnn.csv

pred_ALL9_mobilenetv3.csv

pred_ALL9_vit_resnet.csv

pred_ALL9_swin_densenet.csv

pred_ALL9_efficientnet.csv

pred_ALL9_vgg16.csv

pred_ALL9_vgg19.csv

Per-model confusions (ALL-9): similarly named confusion_ALL9_*.csv in runs/tables/

L1/L2 predictions & confusions: pred_L1_resnet50.csv, confusion_L2_densenet121.csv, etc.

All links are relative so they render in GitHub’s CSV viewer.

11) Reproducibility Notes

AMP is enabled automatically on CUDA (torch.amp.autocast('cuda')).

Early stopping is active; best checkpoints saved to runs/<PHASE>/<MODEL>/best.pt.

Seeds & splits are controlled in train_multilayer.py (see --seed).

For VGG16 training, the classifier head is adapted to match the 224×224 spatial flatten (no shape mismatch).

12) How Inference Works (app.py)

L1 scores Normal vs Abnormal.

If Normal → stop.

If Abnormal → L2 predicts one of 8 classes.

If L2 ∈ {Eczema, Fungal Infections, Pox} → L3 predicts subclass (optional).

The app returns top-k with probabilities + explanation of route.

To run:

streamlit run app.py

13) Push to your old repo under a new branch v2

From the project root:

# if not yet a git repo
git init
git remote add origin <YOUR-OLD-REPO-URL>   # the SkinBench old repo

# create and switch to v2 branch
git checkout -b v2

# keep large checkpoints out of git history (recommended)
echo "runs/**/*.pt" >> .gitignore
echo ".streamlit/" >> .gitignore
echo "__pycache__/" >> .gitignore
git add .
git commit -m "SkinBench v2: multilayer pipeline + evaluations + Streamlit"

# push v2
git push -u origin v2


After the push, open your repo → switch to branch v2 → all the figures under runs/figures/ and CSVs under runs/tables/ will render automatically in GitHub.

14) Citations & Inspiration

(Place your paper citations here; your PDFs are in /papers locally. You can add a short BibTeX list.)

15) License

MIT (or your preferred license).
