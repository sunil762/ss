# Diabetic Retinopathy Detection — Deep Learning Portfolio Project

An end-to-end project: a CNN (transfer learning) classifies retinal fundus
photos into 5 diabetic retinopathy severity grades, served through a live
Streamlit web app with Grad-CAM explainability, with usage logged for a
Power BI analytics dashboard.

> **This is an educational/portfolio project, not a medical device.** It has
> not been clinically validated, is not FDA/CE cleared, and must never be
> used for real diagnosis. The app displays this disclaimer to users too.

## Project structure

```
dr_project/
├── model_training/
│   └── train.py                 # transfer-learning training script (run on Colab/Kaggle GPU)
├── app/
│   ├── app.py                   # Streamlit app: upload image -> prediction + Grad-CAM
│   ├── best_model.pth           # trained weights (you add this after training)
│   ├── predictions_log.csv      # created automatically as the live app is used
│   └── predictions_log_sample.csv  # 180 synthetic rows, so you can build the
│                                    # Power BI dashboard before the app has real usage
├── requirements.txt
├── powerbi_dashboard_guide.md   # step-by-step Power BI build instructions
└── README.md                    # this file
```

## The 5 severity classes

| Class | Label |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

This follows the International Clinical Diabetic Retinopathy (ICDR) severity
scale, the same labeling used by the APTOS 2019 Kaggle dataset.

---

## Step 1 — Get the dataset

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to the [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
3. Download `train.csv` and `train_images/` (fundus photos, ~35,000 total in
   the full ecosystem; the competition training set is ~3,700 labeled images)

## Step 2 — Train the model (Google Colab, free GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com), new notebook
2. **Runtime → Change runtime type → GPU**
3. Upload `model_training/train.py`, or paste its contents into a cell
4. Get the dataset into Colab — either:
   - Upload the zip directly (`Files` panel → drag and drop), or
   - Use the Kaggle API:
     ```python
     !pip install kaggle
     # upload your kaggle.json API token first (Kaggle account -> Create API Token)
     !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/
     !kaggle competitions download -c aptos2019-blindness-detection
     !unzip -q aptos2019-blindness-detection.zip -d data/
     ```
5. Install dependencies: `!pip install torch torchvision scikit-learn pandas matplotlib`
6. Run: `!python train.py`
7. Training takes roughly 20–40 minutes for 15 epochs on a Colab GPU.
   Watch the printed `val_kappa` — quadratic weighted kappa is the standard
   metric for this dataset (agreement between predicted and true severity,
   accounting for how "far off" a wrong prediction is). Above ~0.7 is a
   reasonable portfolio-quality result; the original Kaggle competition's
   top solutions score around 0.93+ with heavy ensembling.
8. Download the resulting `best_model.pth` from Colab

## Step 3 — Wire the trained model into the app

Copy `best_model.pth` into the `app/` folder, next to `app.py`.

## Step 4 — Run it locally to test

```bash
cd dr_project
pip install -r requirements.txt
cd app
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a fundus image, confirm you get a
prediction, a confidence bar chart, and a Grad-CAM heatmap overlay showing
which regions of the retina drove the prediction.

If you skip training and just want to see the UI, it still runs in **demo
mode** with an untrained network — predictions will be meaningless, but the
interface is fully functional so you can iterate on the UI first.

## Step 5 — Deploy it live (Streamlit Community Cloud, free)

1. Push this whole `dr_project` folder to a public GitHub repo
   - **Important:** `best_model.pth` is a large binary file (~16MB for
     EfficientNet-B0). GitHub allows it, but if you outgrow the free repo
     size limits later, use [Git LFS](https://git-lfs.com) for model weights.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. **New app** → select your repo → main file path: `app/app.py` → **Deploy**
4. You'll get a public URL like `https://your-app-name.streamlit.app`

That URL is your live, shareable frontend + backend — no separate server to
manage, since Streamlit Cloud hosts both the UI and the inference logic
together.

## Step 6 — Build the Power BI analytics dashboard

Full instructions in [`powerbi_dashboard_guide.md`](powerbi_dashboard_guide.md).
Short version: every prediction the live app makes gets appended to
`predictions_log.csv`; point Power BI at that file (or the bundled
`predictions_log_sample.csv` to build the dashboard immediately) and follow
the guide to build class-distribution, confidence, and volume-over-time
visuals, then publish it as a shareable Power BI Service link.

---

## Design notes / what to say about this project in an interview

- **Why transfer learning, not training from scratch**: fundus images are a
  small, specialized dataset (thousands, not millions, of images). Starting
  from ImageNet weights and fine-tuning gives far better results with far
  less data and compute than training a CNN from scratch.
- **Why quadratic weighted kappa, not plain accuracy**: DR severity is
  ordinal (0-4), and misclassifying "No DR" as "Proliferative DR" is a much
  worse error than confusing "Mild" and "Moderate." Kappa accounts for that;
  plain accuracy doesn't.
- **Why class-weighted loss**: the dataset is heavily imbalanced (roughly
  half the images are "No DR"). Without weighting, the model can get
  deceptively high accuracy by mostly predicting the majority class.
- **Why Grad-CAM**: for any medical-adjacent ML project, showing *why* the
  model made a prediction (which retinal regions it focused on) is both a
  trust-building UX feature and a way to sanity-check the model isn't
  keying off irrelevant image artifacts (e.g. camera vignetting).

## Extending this further

- Add a second head to also predict image quality/gradability, since real
  DR screening pipelines reject unusable photos before classification
- Ensemble 2-3 backbones (EfficientNet-B0 + ResNet50) and average predictions
  for a meaningful accuracy bump
- Swap local CSV logging for a small hosted database (e.g. Supabase/Postgres)
  so Power BI can connect live instead of via manual CSV re-upload
