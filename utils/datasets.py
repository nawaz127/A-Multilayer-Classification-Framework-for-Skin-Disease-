"""
utils/datasets.py
-----------------
Dataset utilities for SkinBench.

WHAT'S NEW
- Canonical 9-class mapping kept as before.
- Added SUBCLASS phase for fine-grained labels under:
    * Eczema: [Atopic, Seborrheic]
    * Fungal: [Tinea, Candidiasis]
    * Pox   : [Chickenpox, Monkeypox]
- Robust folder-name parsing (case/space/() tolerant).
- Optional keep_paths to return image path for analysis tools.

Returns:
    If keep_paths=False:  (img, y)
    If keep_paths=True :  (img, y, path)
"""

import os, re, numpy as np
from PIL import Image
from torch.utils.data import Dataset

# ------------- Canonical class names (9-way) -------------
LABELS_9 = [
    'Acne',
    'Bacterial/Impetigo',
    'Eczema',
    'Fungal',
    'Normal',
    'Pigmentation',
    'Pox',
    'Psoriasis',
    'Scabies'
]
IDX9 = {c: i for i, c in enumerate(LABELS_9)}

# Map many folder name variants to canonical names
CANONICAL = {
    'acne': 'Acne',

    'bacterial infections (impetigo)': 'Bacterial/Impetigo',
    'bacterial infections': 'Bacterial/Impetigo',
    'impetigo': 'Bacterial/Impetigo',

    'eczema': 'Eczema',
    'atopic dermatitis': 'Eczema',
    'seborrhoeic dermatitis': 'Eczema',

    'fungal infections (tinea, candidiasis)': 'Fungal',
    'fungal infections': 'Fungal',
    'candidiasis': 'Fungal',
    'tinea': 'Fungal',

    'normal': 'Normal',

    'pigmentation disorders': 'Pigmentation',
    'pigmentation': 'Pigmentation',

    'pox': 'Pox',
    'chickenpox': 'Pox',
    'monkeypox': 'Pox',

    'Psoriasis': 'Psoriasis',
    'scabies': 'Scabies',
}

def _canon(name: str) -> str:
    """Normalize a folder/file token then map to canonical class when possible."""
    n = name.lower()
    n = re.sub(r'\(.*?\)', '', n)          # drop (...) hints
    n = re.sub(r'[_\-]', ' ', n)           # unify separators
    n = re.sub(r'\d+.*', '', n).strip()    # strip leading id ranges
    n = re.sub(r'\s+', ' ', n)
    return CANONICAL.get(n, CANONICAL.get(name.lower(), name))

# ------------- SUBCLASS definitions (6-way) -------------
# Order matters (stable label indices for training/eval)
SUBCLASS_LABELS = [
    'Eczema_Atopic',
    'Eczema_Seborrheic',
    'Fungal_Tinea',
    'Fungal_Candidiasis',
    'Pox_Chickenpox',
    'Pox_Monkeypox'
]
SUBIDX = {c: i for i, c in enumerate(SUBCLASS_LABELS)}

# For path scanning, minimal keyword sets per subclass
_SUB_KWS = {
    'Eczema_Atopic':       ['eczema', 'atopic'],
    'Eczema_Seborrheic':   ['eczema', 'seborr'],
    'Fungal_Tinea':        ['fungal', 'tinea'],
    'Fungal_Candidiasis':  ['fungal', 'candid'],
    'Pox_Chickenpox':      ['pox', 'chicken'],
    'Pox_Monkeypox':       ['pox', 'monkey'],
}

def _match_subclass(path_tokens):
    """
    Try to match subclass by scanning path tokens.
    Returns subclass name or None if not matched.
    """
    toks = [t.lower() for t in path_tokens]
    for name, kws in _SUB_KWS.items():
        if all(any(kw in t for t in toks) for kw in kws):
            return name
    return None


class RoutedFolder(Dataset):
    """
    A light wrapper that scans file paths under `root` and assigns labels.

    phase:
      - 'ALL9'     -> 9 classes (LABELS_9)
      - 'L1'       -> 2 classes: [Normal, Abnormal]
      - 'L2'       -> 8 diseases (LABELS_9 without 'Normal')
      - 'SUBCLASS' -> 6 fine-grained subclasses for {Eczema, Fungal, Pox}

    keep_paths:
      - if True, __getitem__ returns (img, y, path) for analysis tools.
    """

    def __init__(self, root, transform=None, phase='ALL9', keep_paths=False):
        self.root = root
        self.transform = transform
        self.phase = phase
        self.keep_paths = keep_paths

        # Collect all image files by walking the tree once
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        files = []
        for d, _, fnames in os.walk(root):
            for f in fnames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(d, f))

        self.samples = []
        if phase == 'ALL9':
            classes = LABELS_9
        elif phase == 'L1':
            classes = ['Normal', 'Abnormal']
        elif phase == 'L2':
            classes = [c for c in LABELS_9 if c != 'Normal']
        elif phase == 'SUBCLASS':
            classes = SUBCLASS_LABELS
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Build samples with label indices
        for p in files:
            parts = [tok for tok in p.split(os.sep) if tok]
            # First, find a canonical 9-class label from any parent folder
            canon = None
            for tok in reversed(parts[:-1]):  # ignore filename
                cname = _canon(tok)
                if cname in IDX9:
                    canon = cname
                    break
            # Fall back to filename or final folder if not found
            if canon is None:
                cname = _canon(parts[-2] if len(parts) >= 2 else parts[-1])
                if cname in IDX9:
                    canon = cname

            # Skip files we cannot reliably map
            if canon is None:
                continue

            if phase == 'ALL9':
                y = IDX9[canon]

            elif phase == 'L1':
                y = 0 if canon == 'Normal' else 1

            elif phase == 'L2':
                if canon == 'Normal':
                    continue  # exclude normals
                l2_classes = [c for c in LABELS_9 if c != 'Normal']
                y = l2_classes.index(canon)

            elif phase == 'SUBCLASS':
                # Only keep images that belong to {Eczema, Fungal, Pox} and match a known subclass
                if canon not in ('Eczema', 'Fungal', 'Pox'):
                    continue
                sub = _match_subclass(parts)
                if sub is None:
                    # Couldn't find a precise subclass keyword – skip to avoid noisy labels
                    continue
                y = SUBIDX[sub]

            self.samples.append((p, y))

        # Expose class names for phase
        self.classes = classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.keep_paths:
            return img, y, path
        return img, y
