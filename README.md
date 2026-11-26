# CardioRisk – Heart Disease Risk Predictor (PWA)

Machine-learning pipeline + installable web app (Android, iOS, Windows, macOS, Linux).
Predictions run **in the browser** (no server, no data leaves the device) and work **offline**.

```
ml/train.py          # trains 5 classifiers, evaluates, exports model + metrics
ml/dataset.csv       # training data
public/              # the whole web app (static) – this is what Vercel serves
  index.html, styles.css, app.js
  manifest.webmanifest, sw.js, icons/   # PWA install + offline support
  model.json         # exported logistic-regression weights
  metrics.json       # evaluation results shown in the app
vercel.json
```

## Run locally
```bash
pip install -r ml/requirements.txt
python ml/train.py                 # regenerates public/model.json + metrics.json
npx serve public                  # or: python -m http.server -d public 3000
```

## Deploy to Vercel
1. Push this folder to GitHub.
2. Vercel → **Add New → Project** → import the repo.
3. Framework preset: **Other**. No build command. Output directory is already set to `public` in `vercel.json`.
4. Deploy. The site is served over HTTPS, which PWAs require.

## Install on devices
- **Android (Chrome):** menu → *Install app*, or use the in-app **Install app** button.
- **iOS (Safari):** Share → *Add to Home Screen*.
- **Windows (Edge/Chrome):** click the install icon in the address bar, or the in-app button.

After retraining or changing any file, bump `VERSION` in `public/sw.js` so installed apps update.

## Evaluation metrics
`train.py` reports accuracy, precision, recall, F1, ROC-AUC and the confusion matrix on a stratified 20 % hold-out set, plus
5-fold × 3-repeat cross-validation, and compares against a majority-class baseline.

### ⚠ Result on the provided dataset
On `hospital_synthetic_dataset` every model scores **~50–56 % (ROC-AUC ≈ 0.5)** — the same as guessing. Feature/label
correlations are all |r| < 0.07, and a label-shuffle test gives the same accuracy as the real labels, meaning the dataset
contains **no learnable signal**. Models such as Random Forest reach ~87 % on the *training* data but that is memorisation;
it does not generalise. The app displays these honest numbers and a low-reliability warning.

To reach 80–90 %, retrain on a dataset where the target actually depends on the features (e.g. the UCI/Cleveland Heart
Disease data, typically ~83–88 %):
```bash
python ml/train.py path/to/real_data.csv   # same columns, or edit FEATURES in train.py
```
The warning disappears automatically when cross-validated accuracy ≥ 80 %.

> Educational tool only. Not a medical device; do not use for diagnosis or treatment decisions.
