# � Crop Disease Detector

A Streamlit application that uses a trained TensorFlow model to detect crop diseases from leaf images and display plant health recommendations.

## ✨ What this project contains

- `app.py` – main Streamlit app with two pages: **Scan Crop** and **View Results**
- `crop_disease_model.h5` – trained TensorFlow/Keras model loaded by `app.py`
- `class_names.json` – model label mapping for the 14 supported classes
- `disease_database.py` – dictionary with 38 disease and healthy entries for diagnosis details
- `requirements.txt` – project dependencies
- `runtime.txt` – Python runtime version
- `check_model.py` – optional utility script that loads `model.h5` and prints category count
- `dataset/` – training/validation image folders used by `app.py` to generate dataset statistics

## 🚀 Features

- Upload a leaf image in JPG or PNG format
- Run inference with `crop_disease_model.h5`
- Show prediction confidence and plant status
- Look up cause, transmission, and treatment recommendations from `disease_database.py`
- Keep a scan history for the current session
- Clear history and reset the upload counter
- Display dataset statistics and charts on the **View Results** page

## 🧠 How the app works

- `app.py` loads the model from `crop_disease_model.h5` using `tf.keras.models.load_model`
- It loads `class_names.json` and maps raw model labels to human-readable disease keys
- `disease_database.py` is used to provide cause, transmission, and rectification text for each diagnosis
- Image uploads are resized to `224x224`, normalized, and predicted by the model
- Session state stores `upload_count`, `history`, and upload form state across navigation

## 📊 Actual counts in this project

- `class_names.json` contains: **14** labels
- `disease_database.py` contains: **38** entries
- `requirements.txt` uses UTF-16 encoding and includes:
  - `streamlit`
  - `tensorflow==2.15.0`
  - `numpy`
  - `pandas`
  - `pillow`
  - `plotly`
  - `matplotlib`
  - `h5py`
- `runtime.txt` specifies: `python-3.11`

## 📁 File usage notes

- `app.py` reads dataset statistics from `dataset/train`
- `app.py` loads `crop_disease_model.h5`; the app does not use `model.h5`
- `check_model.py` is a separate verification utility that currently loads `model.h5`

## ▶️ Run the app

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Streamlit:

```bash
streamlit run app.py
```

## 💡 Notes

- The app has two sidebar menu options:
  - **Scan Crop**: upload images and view diagnosis details
  - **View Results**: see dataset metrics, crop distribution, and session analytics
- The project displays both dataset-level charts and session-level charts using Plotly and Matplotlib

---

**Crop Disease Detector** – built for leaf-level disease scanning and session analytics with a trained TensorFlow model.
