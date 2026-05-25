import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json
import os

from disease_database import disease_database


# ============================================================
# DATASET DISTRIBUTION COUNTER (cached)
# ============================================================
@st.cache_data
def get_dataset_distribution(dataset_path):
    """Count images per class folder and return a clean DataFrame."""
    disease_counts = {}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            disease_counts[class_name] = count
    return pd.DataFrame({
        "Disease": list(disease_counts.keys()),
        "Count": list(disease_counts.values())
    }).sort_values(by="Count", ascending=False)


# Load dataset stats once
DATASET_PATH = "dataset/train"
df_dataset = get_dataset_distribution(DATASET_PATH)
TOTAL_DATASET_IMAGES = int(df_dataset["Count"].sum())

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FONT AWESOME CDN (separate injection to avoid parser issues)
# ============================================================
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">',
    unsafe_allow_html=True
)

# ============================================================
# CLEAN, SIMPLE CSS — DARK THEME, FARMER-FRIENDLY
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ================= GLOBAL ================= */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND (LEAF + DARK OVERLAY) */
.stApp {
    background: linear-gradient(rgba(10,15,40,0.85), rgba(10,15,40,0.9)),
                url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: rgba(8, 12, 30, 0.95) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(74,144,226,0.2);
}
/* Sidebar title */
[data-testid="stSidebar"] h1 {
    font-size: 1.3rem !important;
    color: #ffffff !important;
}

/* Sidebar menu label */
[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6c7bb0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Sidebar options */
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

/* Hover */
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(74,144,226,0.08);
    border-color: rgba(74,144,226,0.3);
}

/* Selected */
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: rgba(74,144,226,0.15);
    border-color: #4a90e2;
}

/* ================= HEADINGS ================= */
h1, h2, h3, h4, h5 {
    color: #ffffff !important;
    font-weight: 600 !important;
}

h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1rem !important; color: #8fa1d0 !important; }

/* ================= GLASS CARDS ================= */
.metric-card, .info-box, .stDataFrame, .stTable {
    background: rgba(22, 30, 69, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 1px solid rgba(74,144,226,0.15);
}

/* Metric Cards */
.metric-card {
    padding: 1.3rem;
    text-align: center;
}

.metric-card .icon {
    font-size: 1.6rem;
}

.metric-card .label {
    color: #8fa1d0;
    font-size: 0.75rem;
    text-transform: uppercase;
}

.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}

/* ================= INFO BOX ================= */
.info-box {
    padding: 1.2rem 1.4rem;
    border-left: 4px solid #4a90e2;
}

.info-box h4 {
    color: #4a90e2 !important;
    font-size: 0.9rem !important;
}

.info-box p {
    color: #d1d5db;
    line-height: 1.6;
}

/* ================= FILE UPLOADER ================= */
[data-testid="stFileUploader"] {
    background: rgba(22, 30, 69, 0.7);
    border: 1px dashed #4a90e2;
    border-radius: 12px;
    padding: 1.8rem;
    transition: 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6bbcff;
    background: rgba(22, 30, 69, 0.85);
}

/* ================= BUTTON ================= */
.stButton>button {
    background: linear-gradient(135deg, #4a90e2, #3b4c8a);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.6rem;
    transition: 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #6bbcff, #4a90e2);
    transform: translateY(-1px);
}

/* ================= TABLE (GLASS + LIGHTER) ================= */
.stDataFrame {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px;
    padding: 8px;
    backdrop-filter: blur(6px);
}

.dataframe {
    background: rgba(10, 15, 40, 0.55) !important;
    border-radius: 10px;
    border: 1px solid rgba(74,144,226,0.12) !important;
    table-layout: auto !important;
}

.dataframe thead th {
    background: rgba(255,255,255,0.07) !important;
    color: #c7d2fe !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 0.5rem !important;
    border-bottom: 1px solid rgba(74,144,226,0.15) !important;
    white-space: normal !important;
}

.dataframe tbody tr {
    transition: background 0.15s ease;
}

.dataframe tbody tr:hover {
    background: rgba(74,144,226,0.1) !important;
}

.dataframe td {
    color: #e8ecf4 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.6rem !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    line-height: 1.5;
    max-width: none !important;
}

/* Accuracy column (3rd) — green highlight */
.dataframe td:nth-child(3) {
    color: #4cd964 !important;
    font-weight: 600;
}

/* Status column (4th) */
.dataframe td:nth-child(4) {
    font-weight: 600;
}

/* Horizontal scroll for wide tables */
[data-testid="stDataFrame"] > div {
    overflow-x: auto !important;
}

/* ================= ALERTS ================= */
.stSuccess {
    background: rgba(76, 217, 100, 0.15) !important;
    border-left: 4px solid #4cd964;
}

.stWarning {
    background: rgba(255, 176, 59, 0.15) !important;
    border-left: 4px solid #ffb03b;
}

.stInfo {
    background: rgba(74, 144, 226, 0.15) !important;
    border-left: 4px solid #4a90e2;
}

/* ================= SCROLLBAR ================= */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #2a3875;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if 'upload_count' not in st.session_state:
    st.session_state.upload_count = 0

if 'history' not in st.session_state:
    st.session_state.history = []

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def clear_history():
    st.session_state.history = []
    st.session_state.upload_count = 0
    st.session_state.uploader_key += 1

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('crop_disease_model.h5')

model = load_model()


with open('class_names.json') as f:
    class_names = json.load(f)

# ============================================================
# RAW LABEL → DATABASE KEY MAPPING
# ============================================================
# The model outputs labels like "Tomato__Tomato_YellowLeaf__Curl_Virus"
# but disease_database uses "Tomato - Yellow Leaf Curl Virus".
# An explicit map is the only reliable way to bridge the two.
LABEL_TO_DB_KEY = {
    'Pepper__bell___Bacterial_spot':                    'Pepper (Bell) - Bacterial Spot',
    'Pepper__bell___healthy':                           'Pepper (Bell) - Healthy',
    'Potato___Early_blight':                            'Potato - Early Blight',
    'Potato___Late_blight':                             'Potato - Late Blight',
    'Potato___healthy':                                 'Potato - Healthy',
    'Tomato_Early_blight':                              'Tomato - Early Blight',
    'Tomato_Late_blight':                               'Tomato - Late Blight',
    'Tomato_Leaf_Mold':                                 'Tomato - Leaf Mold',
    'Tomato_Septoria_leaf_spot':                        'Tomato - Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite':      'Tomato - Spider Mites',
    'Tomato__Target_Spot':                              'Tomato - Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus':            'Tomato - Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus':                      'Tomato - Mosaic Virus',
    'Tomato_healthy':                                   'Tomato - Healthy',
}


def clean_label(raw_label):
    """Convert a raw model label to a clean database key and split into plant/disease."""
    db_key = LABEL_TO_DB_KEY.get(raw_label, None)

    if db_key is None:
        # Fallback: best-effort cleanup for any unmapped label
        cleaned = raw_label.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
        # Collapse multiple spaces and title-case
        db_key = ' '.join(cleaned.split()).title()

    if ' - ' in db_key:
        plant_name, disease_name = db_key.split(' - ', 1)
    else:
        plant_name = db_key
        disease_name = 'Unknown'

    return db_key, plant_name.strip(), disease_name.strip()

# ============================================================
# SIDEBAR — SIMPLE, FRIENDLY LANGUAGE
# ============================================================
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1.5rem 0 1rem 0;'>
        <div style='font-size: 2rem; color: #4cd964;'>
            <i class="fa-solid fa-leaf"></i>
        </div>
        <h1 style='font-size: 1.15rem; margin: 0.5rem 0 0 0; color: #f0f0f0;'>
            Crop Disease Detector
        </h1>
        <p style='color: #8fa1d0; font-size: 0.8rem; margin: 0.3rem 0 0 0;'>
            Identify plant diseases instantly
        </p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Menu",
    ["Scan Crop", "View Results"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; padding: 0.75rem 0; font-size: 0.75rem;'>
        <p style='margin: 0; color: #8fa1d0;'>
            <i class="fa-solid fa-robot" style="margin-right: 0.3rem;"></i> AI Powered
        </p>
        <p style='margin: 0.3rem 0 0 0; color: #5a6a9a;'>
            <i class="fa-solid fa-database" style="margin-right: 0.3rem;"></i> Plant Image Data
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1 — SCAN CROP
# ============================================================
if app_mode == "Scan Crop":
    st.markdown("""
        <h1 style='text-align: center;'>
            <i class="fa-solid fa-magnifying-glass" style="margin-right: 0.5rem;"></i>
            Scan Your Crop
        </h1>
        <p style='text-align: center; color: #8fa1d0; font-size: 1rem; margin-bottom: 2rem;'>
            Upload a photo of a leaf to check for diseases
        </p>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a leaf image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Take a clear photo of a single leaf and upload it here",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.3], gap="large")

        with col1:
            st.markdown("""
                <h3><i class="fa-solid fa-camera" style="margin-right: 0.4rem;"></i> Your Image</h3>
                """, unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("""
                <h3><i class="fa-solid fa-brain" style="margin-right: 0.4rem;"></i> Results</h3>
                """, unsafe_allow_html=True)

            with st.spinner("Checking your crop..."):
                img = image.resize((224, 224))
                img_array = np.array(img)
                if img_array.shape[-1] != 3:
                    img_array = np.stack((img_array,)*3, axis=-1)
                img_array = tf.cast(img_array, tf.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                raw_label = class_names[predicted_class_index]

                # Clean the raw model label → database key + display names
                db_key, plant_name, disease_only = clean_label(raw_label)

                st.session_state.upload_count += 1

                rectification = "No information available."
                cause_val = "Not applicable (healthy plant)"
                transmission_val = "No spread (healthy plant)"
                if db_key in disease_database:
                    details_rec = disease_database[db_key]
                    rectification = details_rec.get('rectification', rectification)
                    raw_cause = details_rec.get('cause', '')
                    raw_trans = details_rec.get('transmission', '')
                    if raw_cause and raw_cause != 'N/A':
                        cause_val = raw_cause
                    if raw_trans and raw_trans != 'N/A':
                        transmission_val = raw_trans

                st.session_state.history.append({
                    'Plant Name': plant_name,
                    'Disease': disease_only,
                    'Accuracy': f"{confidence:.1f}%",
                    'Status': "Healthy" if disease_only.lower() == "healthy" else "Diseased",
                    'Causative Agent': cause_val,
                    'Spread Mode': transmission_val,
                    'Recommendation': rectification
                })

            # --- Show result ---
            is_healthy = disease_only.lower() == 'healthy'

            if is_healthy:
                st.success(f"**Great news!** Your **{plant_name}** looks healthy.")
            else:
                st.warning(f"**Disease found:** {disease_only} on **{plant_name}**")

            st.info(f"**Confidence:** {confidence:.1f}%")

            if db_key in disease_database:
                details = disease_database[db_key]

                cause = details.get('cause', '')
                transmission = details.get('transmission', '')
                rectification = details.get('rectification', '')

                if is_healthy:
                    # Single positive status box for healthy plants
                    healthy_msg = rectification if rectification and rectification != 'N/A' else (
                        f"Your {plant_name} is in great shape. Keep up the good care!"
                    )
                    st.markdown(f"""
                    <div class="info-box">
                        <h4><i class="fa-solid fa-circle-check"></i> Plant Status</h4>
                        <p>{healthy_msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Cause
                    cause_display = cause if cause and cause != 'N/A' else (
                        "No disease detected — the plant is in good condition."
                    )
                    st.markdown(f"""
                    <div class="info-box">
                        <h4><i class="fa-solid fa-virus"></i> What caused it</h4>
                        <p>{cause_display}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Transmission
                    transmission_display = transmission if transmission and transmission != 'N/A' else (
                        "No infection present, so there is no spread risk."
                    )
                    st.markdown(f"""
                    <div class="info-box">
                        <h4><i class="fa-solid fa-wind"></i> How it spreads</h4>
                        <p>{transmission_display}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Rectification
                    rectification_display = rectification if rectification and rectification != 'N/A' else (
                        "No specific treatment needed at this time."
                    )
                    st.markdown(f"""
                    <div class="info-box">
                        <h4><i class="fa-solid fa-kit-medical"></i> What to do</h4>
                        <p>{rectification_display}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("We don't have detailed info for this result yet.")

    # --- History Table ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <h2 style='text-align: center;'>
            <i class="fa-solid fa-clipboard-list" style="margin-right: 0.4rem;"></i>
            Scan History
        </h2>
        <p style='text-align: center; color: #8fa1d0; margin-bottom: 1.5rem;'>
            All your scans from this session
        </p>
        """, unsafe_allow_html=True)

    if len(st.session_state.history) > 0:
        df_history = pd.DataFrame(st.session_state.history)

        # Glass container wrap
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.06);
            padding: 18px;
            border-radius: 12px;
            backdrop-filter: blur(6px);
            border: 1px solid rgba(74,144,226,0.1);
        ">
        """, unsafe_allow_html=True)

        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "Plant Name": st.column_config.TextColumn(
                    "Plant",
                    width="small",
                ),
                "Disease": st.column_config.TextColumn(
                    "Disease",
                    width="medium",
                ),
                "Accuracy": st.column_config.TextColumn(
                    "Accuracy",
                    width="small",
                ),
                "Status": st.column_config.TextColumn(
                    "Status",
                    width="small",
                ),
                "Causative Agent": st.column_config.TextColumn(
                    "Causative Agent",
                    width="medium",
                ),
                "Spread Mode": st.column_config.TextColumn(
                    "Spread Mode",
                    width="large",
                ),
                "Recommendation": st.column_config.TextColumn(
                    "Recommendation",
                    width="large",
                ),
            }
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.columns([1, 4])[0]:
            st.button("Clear History", on_click=clear_history)

        # ============================================================
        # SCAN ANALYTICS
        # ============================================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
            <h2 style='text-align: center;'>
                <i class="fa-solid fa-chart-pie" style="margin-right: 0.4rem;"></i>
                Scan Analytics
            </h2>
            <p style='text-align: center; color: #8fa1d0; margin-bottom: 1.5rem;'>
                Summary of your scan results
            </p>
            """, unsafe_allow_html=True)

        df_analytics = pd.DataFrame(st.session_state.history)
        total_scans = len(df_analytics)
        healthy_total = int((df_analytics['Status'] == 'Healthy').sum())
        diseased_total = int((df_analytics['Status'] == 'Diseased').sum())

        # Clean accuracy to numeric for trend chart
        df_analytics['Accuracy_num'] = (
            df_analytics['Accuracy'].str.replace('%', '').astype(float)
        )

        # --- Metrics row ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans", total_scans)
        m2.metric("Healthy", healthy_total)
        m3.metric("Diseased", diseased_total)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 1: Pie chart + Accuracy trend ---
        chart1, chart2 = st.columns(2, gap="large")

        with chart1:
            st.markdown("""
                <h3><i class="fa-solid fa-chart-pie" style="margin-right: 0.4rem;"></i> Status Distribution</h3>
                """, unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            status_counts = df_analytics['Status'].value_counts()
            status_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax1)
            ax1.set_title("Scan Result Distribution", fontsize=11, color='white')
            ax1.set_ylabel("")
            ax1.tick_params(colors='white', labelsize=9)
            ax1.set_facecolor('none')
            fig1.patch.set_alpha(0)
            for text in ax1.texts:
                text.set_color('white')
            plt.tight_layout()
            st.pyplot(fig1)

        with chart2:
            st.markdown("""
                <h3><i class="fa-solid fa-chart-line" style="margin-right: 0.4rem;"></i> Accuracy Trend</h3>
                """, unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(
                range(1, len(df_analytics) + 1), df_analytics['Accuracy_num'],
                marker='o', linewidth=2, markersize=6
            )
            ax2.set_title("Accuracy Over Scans", fontsize=11, color='white')
            ax2.set_xlabel("Scan Number", fontsize=9, color='white')
            ax2.set_ylabel("Accuracy (%)", fontsize=9, color='white')
            ax2.tick_params(colors='white', labelsize=8)
            ax2.set_facecolor('none')
            fig2.patch.set_alpha(0)
            for spine in ax2.spines.values():
                spine.set_color('#2a3875')
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 2: Top diseases ---
        st.markdown("""
            <h3><i class="fa-solid fa-ranking-star" style="margin-right: 0.4rem;"></i> Disease Distribution</h3>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background: linear-gradient(145deg, #0f172a, #1e293b);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.35);
            margin-top: 10px;
        ">
        """, unsafe_allow_html=True)

        # Clean folder names for display using LABEL_TO_DB_KEY
        df_chart = df_dataset.copy()
        df_chart['Disease'] = df_chart['Disease'].map(
            lambda x: LABEL_TO_DB_KEY.get(x, x).split(' - ', 1)[-1]
        )

        fig_dist = px.bar(
            df_chart,
            x="Disease",
            y="Count",
            color="Count",
            color_continuous_scale="Tealgrn"
        )

        fig_dist.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=11),
            xaxis_title="",
            yaxis_title="Number of Images",
            xaxis=dict(tickangle=-25),
            coloraxis_showscale=False
        )

        fig_dist.update_traces(
            texttemplate='%{y}',
            textposition='outside',
            marker_line_width=0
        )

        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No scans yet. Upload a leaf image above to get started!")


# ============================================================
# PAGE 2 — VIEW RESULTS (ANALYTICS)
# ============================================================
elif app_mode == "View Results":
    st.markdown("""
        <h1 style='text-align: center;'>
            <i class="fa-solid fa-chart-column" style="margin-right: 0.5rem;"></i>
            View Results
        </h1>
        <p style='text-align: center; color: #8fa1d0; font-size: 1rem; margin-bottom: 2rem;'>
            Dataset overview and your session stats
        </p>
        """, unsafe_allow_html=True)

    # --- Count diseased detections in session ---
    diseased_count = sum(
        1 for r in st.session_state.history
        if r['Disease'].lower() != 'healthy'
    )

    # ============================================================
    # 5 METRIC CARDS — SIMPLE LABELS
    # ============================================================
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="icon"><i class="fa-solid fa-images"></i></div>
            <div class="label">Total Images</div>
            <div class="value">{TOTAL_DATASET_IMAGES:,}</div>
            <div class="note note-blue">Training data</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="icon"><i class="fa-solid fa-tags"></i></div>
            <div class="label">Disease Types</div>
            <div class="value">14</div>
            <div class="note note-blue">Recognized</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
            <div class="label">Uploads</div>
            <div class="value">{st.session_state.upload_count}</div>
            <div class="note note-blue">This session</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="icon" style="color: #4cd964;"><i class="fa-solid fa-bullseye"></i></div>
            <div class="label">Accuracy</div>
            <div class="value">95.28%</div>
            <div class="note note-green">Verified</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="icon" style="color: #ff6b6b;"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div class="label">Detected Cases</div>
            <div class="value">{diseased_count}</div>
            <div class="note note-red">Diseased found</div>
        </div>""", unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ============================================================
    # CHARTS — ONLY TOMATO, POTATO, PEPPER
    # ============================================================
    col_chart1, col_chart2 = st.columns(2, gap="large")

    with col_chart1:
        st.markdown("""
            <h3><i class="fa-solid fa-wheat-awn" style="margin-right: 0.4rem;"></i> Images per Crop</h3>
            """, unsafe_allow_html=True)

        # Build crop-level totals from real dataset
        df_crop_totals = df_dataset.copy()
        df_crop_totals['Crop'] = df_crop_totals['Disease'].map(
            lambda x: LABEL_TO_DB_KEY.get(x, x).split(' - ', 1)[0]
        )
        df_crop_agg = df_crop_totals.groupby('Crop', as_index=False)['Count'].sum().sort_values(by='Count', ascending=False)

        fig_bar = px.bar(
            df_crop_agg, x='Crop', y='Count',
            color='Crop',
            color_discrete_map={
                'Tomato': '#ff6b6b',
                'Potato': '#f0c040',
                'Pepper (Bell)': '#4cd964'
            },
            template='plotly_dark'
        )

        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Images",
            font=dict(family="Inter, sans-serif", size=13, color='#d1d5db'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )

        fig_bar.update_traces(
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1,
            texttemplate='%{y:,}',
            textposition='outside',
            textfont=dict(color='#8fa1d0', size=12)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.markdown("""
            <h3><i class="fa-solid fa-heart-pulse" style="margin-right: 0.4rem;"></i> Crop Health Overview</h3>
            """, unsafe_allow_html=True)

        # Compute healthy vs diseased from real dataset
        healthy_classes = [k for k, v in LABEL_TO_DB_KEY.items() if 'healthy' in k.lower()]
        healthy_images = int(df_dataset[df_dataset['Disease'].isin(healthy_classes)]['Count'].sum())
        diseased_images = TOTAL_DATASET_IMAGES - healthy_images

        df_health = pd.DataFrame({
            'Status': ['Healthy', 'Diseased'],
            'Count': [healthy_images, diseased_images]
        })

        fig_pie = px.pie(
            df_health, names='Status', values='Count',
            hole=0.55,
            color='Status',
            color_discrete_map={'Healthy': '#4cd964', 'Diseased': '#ff6b6b'},
            template='plotly_dark'
        )

        fig_pie.update_layout(
            font=dict(family="Inter, sans-serif", size=13, color='#d1d5db'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15,
                xanchor="center", x=0.5,
                font=dict(color='#8fa1d0', size=13)
            )
        )

        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=14, color='white'),
            marker=dict(line=dict(color='#0f1535', width=2))
        )

        st.plotly_chart(fig_pie, use_container_width=True)


    # ============================================================
    # SESSION STATS (only if there's history)
    # ============================================================
    if len(st.session_state.history) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
            <h2 style='text-align: center;'>
                <i class="fa-solid fa-chart-line" style="margin-right: 0.4rem;"></i>
                Your Session
            </h2>
            <p style='text-align: center; color: #8fa1d0; margin-bottom: 1.5rem;'>
                Summary of what you've scanned so far
            </p>
            """, unsafe_allow_html=True)

        df_session = pd.DataFrame(st.session_state.history)
        healthy_count = sum(1 for d in df_session['Disease'] if d.lower() == 'healthy')
        diseased_session = len(df_session) - healthy_count

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon"><i class="fa-solid fa-seedling"></i></div>
                <div class="label">Plants Scanned</div>
                <div class="value">{df_session['Plant Name'].nunique()}</div>
                <div class="note note-blue">Unique crops</div>
            </div>""", unsafe_allow_html=True)

        with s2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon"><i class="fa-solid fa-microscope"></i></div>
                <div class="label">Diseases Found</div>
                <div class="value">{df_session['Disease'].nunique()}</div>
                <div class="note note-blue">Unique types</div>
            </div>""", unsafe_allow_html=True)

        with s3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon" style="color: #4cd964;"><i class="fa-solid fa-circle-check"></i></div>
                <div class="label">Healthy</div>
                <div class="value">{healthy_count}</div>
                <div class="note note-green">Looking good</div>
            </div>""", unsafe_allow_html=True)

        with s4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon" style="color: #ff6b6b;"><i class="fa-solid fa-circle-exclamation"></i></div>
                <div class="label">Diseased</div>
                <div class="value">{diseased_session}</div>
                <div class="note note-red">Needs attention</div>
            </div>""", unsafe_allow_html=True)

        # Session crop breakdown chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <h3><i class="fa-solid fa-leaf" style="margin-right: 0.4rem;"></i> Scans by Crop</h3>
            """, unsafe_allow_html=True)

        plant_counts = df_session['Plant Name'].value_counts().reset_index()
        plant_counts.columns = ['Plant', 'Count']

        fig_session = px.bar(
            plant_counts, x='Plant', y='Count',
            color='Plant',
            color_discrete_sequence=['#4a90e2', '#4cd964', '#f0c040', '#ff6b6b'],
            template='plotly_dark'
        )

        fig_session.update_layout(
            showlegend=False,
            xaxis_title="", yaxis_title="Scans",
            font=dict(family="Inter, sans-serif", size=13, color='#d1d5db'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )

        fig_session.update_traces(
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1
        )

        st.plotly_chart(fig_session, use_container_width=True)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("No session data yet. Go to **Scan Crop** and upload some images!")


# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0;'>
        <p style='color: #8fa1d0; font-size: 0.9rem; margin: 0;'>
            <i class="fa-solid fa-leaf" style="margin-right: 0.3rem;"></i>
            Crop Disease Detector
        </p>
        <p style='color: #5a6a9a; font-size: 0.78rem; margin: 0.3rem 0 0 0;'>
            <i class="fa-solid fa-robot" style="margin-right: 0.3rem;"></i> AI Powered
            &middot;
            <i class="fa-solid fa-database" style="margin-right: 0.3rem;"></i> Plant Image Data
        </p>
        <p style='color: #5a6a9a; font-size: 0.75rem; margin: 0.5rem 0 0 0;'>
            &copy; 2026 All Rights Reserved
        </p>
        <p style='color: #4a90e2; font-size: 0.75rem; margin: 0.3rem 0 0 0;'>
            Developed with <i class="fa-solid fa-heart" style="color: #ff6b6b;"></i>
            by Awowole Hammad Olamilekan &middot; FTP/CSC/25/0118830
        </p>
    </div>
    """, unsafe_allow_html=True)
