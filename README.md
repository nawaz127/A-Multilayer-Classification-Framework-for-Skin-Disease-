🧬 A Multilayer Classification Framework for Skin Disease Detection on SkinBench (v2)






SkinBench-v2 delivers a three-level (multilayer) classification pipeline for skin disease recognition:

L1 (Binary): Normal vs. Disease

L2 (Coarse multi-class): 8 disease super-classes

ALL9 (Fine multi-class): 9 classes (Normal + 8 diseases)

The repo includes trained evaluation utilities, figures (confusion matrices, ROC/PR curves, reliability diagrams), and tabular outputs. This branch documents reproducible evaluation and a Streamlit demo using the best models you trained.

📂 Repository Structure (v2)
SkinBench/
├─ data_raw/                         # <your dataset>  train/val/test/<Class>/*
├─ eval_tools/
│  ├─ run_all_evals.py               # unified evaluation entrypoint
│  ├─ roc_pr_curves.py               # plotting ROC & PR
│  └─ ...
├─ models/                           # model definitions (resnet50, densenet121, ...)
├─ runs/
│  ├─ ALL9/                          # fine 9-class results
│  ├─ L1/                            # binary results
│  ├─ L2/                            # 8-class results
│  ├─ SUBCLASS/                      # (optional) subclass heads
│  ├─ figures/                       # 🔎 all plots live here (GitHub auto-renders)
│  └─ tables/                        # 📊 all CSVs (confusions, predictions, comparison)
├─ app.py                            # Streamlit demo
├─ README.md                         # this file
└─ .gitignore                        # ignore checkpoints & raw data


Important: Only relative paths are used below (e.g., runs/figures/confusion_ALL9_vit_resnet.png).
Once you push to GitHub on v2, the images and tables render automatically.

🧪 Dataset Layout

Place your dataset under data_raw/ with the standard ImageFolder layout:

data_raw/
├─ train/
│  ├─ Acne/ Bacterial Infections/ Eczema/ Fungal Infections/ Normal/
│  ├─ Pigmentation Disorders/ Pox/ Psoriasis/ Scabies/
├─ val/
│  └─ (same 9 subfolders)
└─ test/
   └─ (same 9 subfolders)

🖥️ Environment
# (conda recommended)
conda create -n torch_gpu python=3.10 -y
conda activate torch_gpu
pip install -r requirements.txt   # create if not present; include torch/torchvision, scikit-learn, matplotlib, pandas, streamlit, etc.

🚀 Training (step-by-step)

You already have trained checkpoints under runs/<PHASE>/<MODEL>/best.pt.
If you want to retrain, run your project’s training entrypoint per phase/model (adapt the command to your script names):

# Example commands (adapt to your train script/args):
# ALL9 (9-class)
python -m models.train --phase ALL9 --model mobilenetv3 --data_dir data_raw --epochs 50 --out runs/ALL9/mobilenetv3

python -m models.train --phase ALL9 --model vit_resnet  --data_dir data_raw --epochs 50 --out runs/ALL9/vit_resnet

python -m models.train --phase ALL9 --model swin_densenet --data_dir data_raw --epochs 50 --out runs/ALL9/swin_densenet

python -m models.train --phase ALL9 --model efficientnet --data_dir data_raw --epochs 50 --out runs/ALL9/efficientnet

python -m models.train --phase ALL9 --model vgg16 --data_dir data_raw --epochs 50 --out runs/ALL9/vgg16

python -m models.train --phase ALL9 --model vgg19 --data_dir data_raw --epochs 50 --out runs/ALL9/vgg19

# (cnn baseline likewise)

# L1 (binary: Normal vs Disease)

python -m models.train --phase L1 --model resnet50 --data_dir data_raw --epochs 30 --out runs/L1/resnet50

# L2 (8-class diseases only)

python -m models.train --phase L2 --model resnet50     --data_dir data_raw --epochs 40 --out runs/L2/resnet50

python -m models.train --phase L2 --model densenet121  --data_dir data_raw --epochs 40 --out runs/L2/densenet121


Tip: Keep checkpoint filenames as runs/<PHASE>/<MODEL>/best.pt so the evaluator auto-discovers them.

📈 Reproducible Evaluation
1) Run everything at once
python -m eval_tools.run_all_evals --data_dir data_raw --phases ALL9 L1 L2 --models resnet50 densenet121 mobilenetv3 cnn vit_resnet swin_densenet efficientnet vgg16 vgg19

Outputs:

Figures → runs/figures/*.png

Tables → runs/tables/*.csv

Comparison table → runs/tables/comparison_all_models.csv

2) Notable notes

Binary ROC/PR on L1: the evaluator expects multi-class probabilities for OVR plots; L1 uses binary heads. The script now skips OVR curves for L1 to avoid IndexError and uses binary ROC/PR instead.

Your earlier issues with RoutedFolder were resolved by passing the correct constructor args; evaluator now builds loaders internally.

🧪 Results (your reported metrics)
ALL9 (9 classes) — Test set
Model	Accuracy	Macro-F1
ViT-ResNet	0.9843	0.9836
Swin-DenseNet	0.9843	0.9839
EfficientNet	0.9833	0.9827
MobileNetV3	0.9815	0.9808
VGG16	0.9721	0.9713
VGG19	0.9667	0.9668
CNN baseline	0.6389	0.6220
ResNet50	(no ALL9 checkpoint)	–
DenseNet121	(no ALL9 checkpoint)	–

Key plots (auto-rendered on GitHub):

Confusion matrices
runs/figures/confusion_ALL9_vit_resnet.png
runs/figures/confusion_ALL9_swin_densenet.png
runs/figures/confusion_ALL9_efficientnet.png
runs/figures/confusion_ALL9_mobilenetv3.png
runs/figures/confusion_ALL9_vgg16.png
runs/figures/confusion_ALL9_vgg19.png
runs/figures/confusion_ALL9_cnn.png

ROC (one-vs-rest):
runs/figures/roc_ovr_ALL9_vit_resnet.png
runs/figures/roc_ovr_ALL9_swin_densenet.png
runs/figures/roc_ovr_ALL9_efficientnet.png
runs/figures/roc_ovr_ALL9_mobilenetv3.png
runs/figures/roc_ovr_ALL9_vgg16.png
runs/figures/roc_ovr_ALL9_vgg19.png
runs/figures/roc_ovr_ALL9_cnn.png

PR (one-vs-rest):
runs/figures/pr_ovr_ALL9_vit_resnet.png
runs/figures/pr_ovr_ALL9_swin_densenet.png
runs/figures/pr_ovr_ALL9_efficientnet.png
runs/figures/pr_ovr_ALL9_mobilenetv3.png
runs/figures/pr_ovr_ALL9_vgg16.png
runs/figures/pr_ovr_ALL9_vgg19.png
runs/figures/pr_ovr_ALL9_cnn.png

Reliability diagrams:
runs/figures/reliability_ALL9_vit_resnet.png
runs/figures/reliability_ALL9_swin_densenet.png
runs/figures/reliability_ALL9_efficientnet.png
runs/figures/reliability_ALL9_mobilenetv3.png
runs/figures/reliability_ALL9_vgg16.png
runs/figures/reliability_ALL9_vgg19.png
runs/figures/reliability_ALL9_cnn.png

L1 (Binary: Normal vs Disease) — Test set
Model	Accuracy	Macro-F1
ResNet50	0.9980	0.9959

Plots:
runs/figures/confusion_L1_resnet50.png
runs/figures/roc_binary_L1_resnet50.png
runs/figures/pr_binary_L1_resnet50.png
runs/figures/reliability_L1_resnet50.png

L2 (8 diseases) — Test set
Model	Accuracy	Macro-F1
DenseNet121	0.9853	0.9849
ResNet50	0.9827	0.9824

Plots:
runs/figures/confusion_L2_densenet121.png
runs/figures/confusion_L2_resnet50.png
runs/figures/roc_ovr_L2_densenet121.png
runs/figures/roc_ovr_L2_resnet50.png
runs/figures/pr_ovr_L2_densenet121.png
runs/figures/pr_ovr_L2_resnet50.png
runs/figures/reliability_L2_densenet121.png
runs/figures/reliability_L2_resnet50.png

CSV artifacts (render on GitHub)

Model comparison: runs/tables/comparison_all_models.csv

Per-model confusion tables (CSV):
runs/tables/confusion_ALL9_vit_resnet.csv, ..._swin_densenet.csv, ..._efficientnet.csv, ..._mobilenetv3.csv, ..._vgg16.csv, ..._vgg19.csv, ..._cnn.csv
runs/tables/confusion_L1_resnet50.csv, runs/tables/confusion_L2_resnet50.csv, runs/tables/confusion_L2_densenet121.csv

Predictions (CSV per model):
runs/tables/pred_ALL9_vit_resnet.csv, ..._swin_densenet.csv, ..._efficientnet.csv, ..._mobilenetv3.csv, ..._vgg16.csv, ..._vgg19.csv, ..._cnn.csv
runs/tables/pred_L1_resnet50.csv, runs/tables/pred_L2_resnet50.csv, runs/tables/pred_L2_densenet121.csv

🌐 Streamlit Demo
streamlit run app.py


Features:

Upload an image → L1 → L2 → ALL9 cascade prediction

Uses your best checkpoints from runs/<PHASE>/<MODEL>/best.pt

Supports subclass prediction hooks (optional SUBCLASS/ heads)

If you see a Missing key(s) in state_dict error, it means the checkpoint was trained with a different head/backbone name. Re-export your model with the same class names and module keys used in models/*.py for inference.

🔒 Keeping the repo clean (no heavy files)

Add to .gitignore (already recommended):

# checkpoints
runs/**/best.pt
runs/**/best.pth
runs/**/last.pt
runs/**/last.pth
runs/**/checkpoints/
models/**/best*.pt
models/**/best*.pth

# raw data
data_raw/

# python cache
__pycache__/
*.pyc
*.pyo


If you accidentally committed large checkpoints:

git rm --cached -r runs models
git commit -m "remove tracked checkpoints"
git push origin v2

🧩 Reproducing exactly these plots

All the images shown above are already produced under runs/figures/.
To regenerate them after retraining, simply re-run:

python -m eval_tools.run_all_evals --data_dir data_raw --phases ALL9 L1 L2 --models resnet50 densenet121 mobilenetv3 cnn vit_resnet swin_densenet efficientnet vgg16 vgg19

📜 License

This repository is released under the MIT License. See LICENSE for details.

🙌 Acknowledgements

Thanks to all open-source authors of PyTorch, Torchvision, scikit-learn, Matplotlib, and the SkinBench dataset contributors. Your evaluation plots and tables (runs/figures, runs/tables) make the results fully auditable on GitHub.
