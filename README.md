# Liver Cirrhosis Stage Prediction

A Streamlit app and training notebook to predict liver cirrhosis stage using clinical features.

## Project Structure

```
liver_stage_prediction/
│── data/
│   └── liver_cirrhosis.csv
│── train.ipynb
│── model.pkl
│── features.json
│── app.py
│── requirements.txt
│── README.md
```

> **Note:** `features.json` is generated after training. A pre-trained `model.pkl` and `features.json` are already included so the app runs out-of-the-box.

## Setup

1. **Unzip** this project.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Train the Model

Open the notebook and run all cells:

```bash
jupyter notebook train.ipynb
```

This will read `data/liver_cirrhosis.csv`, train a Random Forest pipeline, and update `model.pkl` and `features.json`.

## Run the App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal.

## Using Your Own Dataset

- Place your CSV at `data/liver_cirrhosis.csv`.
- Ensure it contains a target column named **Stage** (values like `Mild`, `Moderate`, `Severe`).  
  If **Stage** is missing, the notebook will attempt to create a proxy target so you can still train.
- Numeric columns are auto-imputed with median; categorical columns are one-hot encoded.

## Notes

- The app dynamically builds the prediction form based on `features.json`.  
- If you change columns or add new ones, **re-run the notebook** to regenerate the model and features.
