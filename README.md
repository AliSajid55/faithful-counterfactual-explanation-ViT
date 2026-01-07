# faithful-counterfactual-explanation-ViT
Explainable Artificial Intelligence (XAI) focuses on making deep learning models transparent and trustworthy by explaining their decisions. FCET investigates attention heads-level counterfactual explanations for Vision Transformers using internal representation manipulation, MC/MI analysis, and contrastive masking.

## Notebook 1 — ViT Classifier Fine-Tuning (CUB-200)

This notebook trains a **Vision Transformer (ViT-B/16, 224×224)** classifier on the **CUB-200-2011** dataset to obtain a strong backbone model for downstream **faithful counterfactual explanation** experiments.

### What this notebook does
- Installs and uses `timm==0.9.10` and loads **`vit_base_patch16_224`** (pretrained).
- Parses official CUB metadata files (`images.txt`, `classes.txt`, `image_class_labels.txt`, `train_test_split.txt`).
- Builds a reproducible **train/val/test** setup:
  - Uses the official train/test split from CUB
  - Creates a **stratified validation split (15%)** from the train set
- Implements a full fine-tuning pipeline with:
  - ViT-friendly image transforms (RandomResizedCrop, ColorJitter, Normalize)
  - **AdamW** optimizer + **warmup → cosine LR schedule**
  - **AMP (mixed precision)** training for speed
  - **Early stopping** based on validation accuracy
  - Logging of train/val loss, accuracy, and timings per epoch
- Evaluates the best checkpoint on the test set:
  - Test Top-1 Accuracy
  - Confusion Matrix (CSV + PNG)
  - Per-class accuracy CSV and top confusions summary

### Key configuration (main defaults)
- Model: `vit_base_patch16_224` (pretrained, `num_classes=200`)
- Image size: `224`
- Epochs: `100` (with early stopping)
- Validation split: `15%` (stratified)
- Loss: CrossEntropy with label smoothing (`label_smoothing=0.1`)
- Optimizer: AdamW (`lr=3e-4`, `weight_decay=5e-2`)
- Scheduler: linear warmup (`5 epochs`) → cosine decay
- Batch size: `32` (adjustable)

### Inputs
- Dataset path (Kaggle):
  - `DATA_ROOT = "/kaggle/input/cub2002011/CUB_200_2011"`

### Outputs / Artifacts (saved to `/kaggle/working`)
- `cub_meta_all.csv`, `cub_train.csv`, `cub_val.csv`, `cub_test.csv` (reproducible splits)
- `labels.txt` (class names)
- `vit_cls_best.pt` (best model checkpoint by val accuracy)
- `train_log.csv` (epoch-wise metrics)
- `meta_cls.json` (model + preprocessing metadata)
- `confusion_matrix.csv`, `confusion_matrix.png`
- `per_class_accuracy.csv`
- Phase-1 ZIP export (optional packaging step)

### How to run
Run cells in order:
1. **Cell 1:** Setup + parse CUB + split + save CSVs
2. **Cell 2:** Transforms + Dataset + DataLoaders
3. **Cell 3:** Train ViT (AMP) + Save best + Logs + Meta
4. **Cell 4:** Test evaluation + confusion matrix + per-class stats
5. **Cell 5:** (Optional) Create ZIP for Phase-1 artifacts

## Notebook 2 — Autoencoder Fine-Tuning for Latent Reconstruction

This notebook trains an **autoencoder (Encoder–Decoder)** aligned with the fine-tuned ViT classifier to enable **latent-space reconstruction**, which is a prerequisite for generating **faithful counterfactual explanations** through internal representation manipulation.

### What this notebook does
- Loads the **pretrained ViT classifier backbone** from Notebook 1.
- Extracts intermediate **latent representations** suitable for reconstruction.
- Defines an **autoencoder architecture** that learns to reconstruct input images from these internal features.
- Trains the autoencoder while **keeping the classifier behavior intact**, ensuring that reconstructions remain faithful to the original decision process.
- Tracks reconstruction quality across training and validation splits.

### Role in the FCET pipeline
This stage establishes a **stable and faithful reconstruction mechanism**, allowing later notebooks to:
- Modify internal representations (MC/MI components)
- Decode them back into the image space
- Observe controlled, explanation-driven changes without arbitrary input perturbations

### Key configuration
- Input: latent features extracted from the ViT encoder
- Reconstruction target: original input image
- Loss function: **Reconstruction loss (L1 / L2)**  
- Optimizer: Adam / AdamW
- Training strategy:
  - Frozen or partially frozen classifier
  - Autoencoder trained independently for reconstruction fidelity

### Inputs
- Trained ViT checkpoint from Notebook 1  
  - e.g. `vit_cls_best.pt`
- Dataset splits (same CUB-200 splits as Notebook 1)

### Outputs / Artifacts
- Trained autoencoder weights (encoder + decoder)
- Reconstruction samples (original vs reconstructed images)
- Training logs (loss curves)
- Serialized metadata for downstream counterfactual experiments

### How to run
Run cells sequentially:
1. **Cell 1:** Load ViT classifier and dataset
2. **Cell 2:** Define encoder–decoder architecture
3. **Cell 3:** Train autoencoder on latent features
4. **Cell 4:** Evaluate reconstruction quality and save checkpoints

## Notebook 3 — MC/MI Component Analysis in Vision Transformers

This notebook identifies **Most Contributing (MC)** and **Most Influential (MI)** components within the Vision Transformer by analyzing the impact of internal representation changes on model predictions. These components form the foundation for generating **faithful counterfactual explanations** in subsequent stages.

### What this notebook does
- Loads the **fine-tuned ViT classifier** and the **trained autoencoder** from previous notebooks.
- Extracts internal ViT representations (token-level / attention-level features).
- Quantifies the contribution and influence of internal components by measuring:
  - Prediction confidence changes
  - Class logit sensitivity under controlled perturbations
- Separates components into:
  - **MC (Most Contributing):** components that strongly support the current prediction
  - **MI (Most Influential):** components whose modification can significantly alter the prediction
- Stores MC/MI indices and statistics for reproducible counterfactual manipulation.

### Role in the FCET pipeline
This stage provides a **faithfulness guarantee** by ensuring that counterfactual explanations are generated through **causal internal components**, rather than arbitrary or visually implausible input perturbations.

Specifically, it enables:
- Targeted suppression of MC components
- Selective activation or modification of MI components
- Controlled internal interventions aligned with model reasoning

### Key configuration
- Analysis space: internal ViT tokens / attention representations
- Metrics:
  - Logit change
  - Prediction confidence variation
- Selection strategy:
  - Threshold- or rank-based MC/MI identification
- Model behavior preserved outside selected components

### Inputs
- ViT classifier checkpoint (from Notebook 1)
- Autoencoder model (from Notebook 2)
- Evaluation samples from CUB-200 dataset

### Outputs / Artifacts
- Serialized MC/MI component indices
- Component-wise importance scores
- Saved statistics for downstream counterfactual generation
- Optional visual summaries of influential components

### How to run
Execute cells in order:
1. **Cell 1:** Load trained models and dataset
2. **Cell 2:** Extract internal ViT representations
3. **Cell 3:** Compute MC and MI component scores
4. **Cell 4:** Save MC/MI metadata for counterfactual masking

## Notebook 4 — Contrastive Masking & Counterfactual Generation

This notebook performs **contrastive masking and counterfactual generation** by manipulating the internally identified **MC (Most Contributing)** and **MI (Most Influential)** components to produce **faithful, class-flipping counterfactual explanations** for Vision Transformer predictions.

### What this notebook does
- Loads the **ViT classifier**, **autoencoder**, and **MC/MI metadata** from previous notebooks.
- Applies **contrastive masking** to isolate the minimal internal components required to preserve the original prediction.
- Generates **counterfactual representations** by:
  - Suppressing selected MC components
  - Activating or amplifying selected MI components
- Verifies counterfactual validity by confirming a **prediction flip with high confidence**.
- Decodes modified internal representations back to the image space using the trained autoencoder.

### Role in the FCET pipeline
This stage completes the FCET framework by transforming **internal, causally meaningful interventions** into **visually interpretable counterfactual images**, ensuring that explanations are:
- **Faithful:** derived from true decision-driving components
- **Minimal:** based on the smallest necessary internal changes
- **Plausible:** visually coherent and semantically consistent

### Key operations
- MC masking: retains only components necessary for the original class
- MI activation: introduces minimal internal changes to induce class flip
- Counterfactual verification via classifier confidence
- Image reconstruction via decoder

### Inputs
- ViT classifier checkpoint
- Trained autoencoder
- MC/MI component metadata (Notebook 3)
- Evaluation images from CUB-200 dataset

### Outputs / Artifacts
- Generated counterfactual images (original vs counterfactual)
- Confidence scores before and after intervention
- Counterfactual success statistics
- Optional visual comparisons for qualitative analysis

### How to run
Execute cells sequentially:
1. **Cell 1:** Load models and MC/MI metadata
2. **Cell 2:** Apply contrastive MC masking
3. **Cell 3:** Activate MI components and generate counterfactual features
4. **Cell 4:** Decode and evaluate counterfactual images
