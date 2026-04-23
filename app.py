import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

from disease_database import disease_database

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
# CLEAN, SIMPLE CSS — DARK THEME, FARMER-FRIENDLY
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp, .main {
        background-color: #0f1535 !important;
        color: #f0f0f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0f28 !important;
        border-right: 1px solid #1e2756 !important;
    }

    [data-testid="stSidebar"] .stRadio > label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #8fa1d0;
        letter-spacing: 0.03em;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        background: transparent;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        border: 1px solid transparent;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: #161e45;
        border-color: #2a3875;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
        text-shadow: none !important;
        font-weight: 600 !important;
    }

    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; color: #8fa1d0 !important; }

    /* Metric Cards */
    .metric-card {
        background-color: #161e45;
        padding: 1.25rem 1rem;
        border-radius: 10px;
        border: 1px solid #2a3875;
        margin-bottom: 1rem;
        text-align: center;
    }

    .metric-card .icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }

    .metric-card .label {
        color: #8fa1d0;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }

    .metric-card .value {
        color: #f0f0f0;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .metric-card .note {
        font-size: 0.78rem;
        font-weight: 500;
    }

    .note-green  { color: #4cd964; }
    .note-blue   { color: #4a90e2; }
    .note-red    { color: #ff6b6b; }

    /* Info Boxes */
    .info-box {
        background-color: #161e45;
        padding: 1.25rem 1.5rem;
        border-left: 4px solid #4a90e2;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }

    .info-box h4 {
        color: #4a90e2 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.4rem !important;
    }

    .info-box p {
        color: #d1d5db;
        line-height: 1.6;
        margin: 0;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2a3875;
        color: #f0f0f0;
        border: 1px solid #3b4c8a;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #3b4c8a;
        border-color: #4a90e2;
        color: #ffffff;
        transform: none !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #161e45;
        border: 1px dashed #3b4c8a;
        border-radius: 8px;
        padding: 1.5rem;
    }

    /* Table */
    .dataframe {
        background-color: #161e45 !important;
        border: 1px solid #2a3875 !important;
        border-radius: 6px;
    }

    .dataframe thead tr th {
        background-color: #0b0f28 !important;
        color: #8fa1d0 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #2a3875 !important;
        padding: 0.8rem !important;
    }

    .dataframe tbody tr {
        background-color: transparent !important;
        border-bottom: 1px solid #1e2756 !important;
    }

    .dataframe tbody tr:hover {
        background-color: #1e2756 !important;
    }

    .dataframe tbody tr td {
        color: #f0f0f0 !important;
        padding: 0.8rem !important;
    }

    .stSuccess, .stInfo, .stWarning {
        background-color: #161e45;
    }

    hr { border-top: 1px solid #1e2756; }
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
    return tf.keras.models.load_model('model.h5')

model = load_model()

# ============================================================
# CLASS NAMES — ALL 38 (model output layer needs all of them)
# ============================================================
class_names = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy', 'Corn - Cercospora Leaf Spot',
    'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy', 'Grape - Black Rot',
    'Grape - Esca (Black Measles)', 'Grape - Leaf Blight', 'Grape - Healthy', 'Orange - Haunglongbing (Greening)',
    'Peach - Bacterial Spot', 'Peach - Healthy', 'Pepper (Bell) - Bacterial Spot', 'Pepper (Bell) - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy',
    'Soybean - Healthy', 'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold',
    'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot',
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
]


# ============================================================
# SIDEBAR — SIMPLE, FRIENDLY LANGUAGE
# ============================================================
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1.5rem 0 1rem 0;'>
        <div style='font-size: 2.2rem;'>🌿</div>
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
    ["🔍 Scan Crop", "📊 View Results"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; padding: 0.75rem 0; font-size: 0.75rem;'>
        <p style='margin: 0; color: #8fa1d0;'>🤖 AI Powered</p>
        <p style='margin: 0.3rem 0 0 0; color: #5a6a9a;'>Plant Image Data</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1 — SCAN CROP
# ============================================================
if app_mode == "🔍 Scan Crop":
    st.markdown("""
        <h1 style='text-align: center;'>🔍 Scan Your Crop</h1>
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
            st.markdown("### 📸 Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("### 🧠 Results")

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
                disease_name = class_names[predicted_class_index]

                if ' - ' in disease_name:
                    plant_name, disease_only = disease_name.split(' - ', 1)
                else:
                    plant_name = disease_name
                    disease_only = "Unknown"

                st.session_state.upload_count += 1

                rectification = "No information available."
                if disease_name in disease_database:
                    rectification = disease_database[disease_name]['rectification']

                st.session_state.history.append({
                    'S/N': st.session_state.upload_count,
                    'Plant Name': plant_name,
                    'Disease': disease_only,
                    'Recommendation': rectification
                })

            # --- Show result ---
            is_healthy = disease_only.lower() == 'healthy'

            if is_healthy:
                st.success(f"✅ **Great news!** Your **{plant_name}** looks healthy.")
            else:
                st.warning(f"⚠️ **Disease found:** {disease_only} on **{plant_name}**")

            st.info(f"🎯 **Confidence:** {confidence:.1f}%")

            if disease_name in disease_database:
                details = disease_database[disease_name]

                st.markdown(f"""
                <div class="info-box">
                    <h4>🦠 What caused it</h4>
                    <p>{details['cause']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                    <h4>🌬️ How it spreads</h4>
                    <p>{details['transmission']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="info-box">
                    <h4>💊 What to do</h4>
                    <p>{details['rectification']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("We don't have detailed info for this result yet.")

    # --- History Table ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <h2 style='text-align: center;'>📋 Scan History</h2>
        <p style='text-align: center; color: #8fa1d0; margin-bottom: 1.5rem;'>
            All your scans from this session
        </p>
        """, unsafe_allow_html=True)

    if len(st.session_state.history) > 0:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True, hide_index=True, height=350)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.columns([1, 3])[0]:
            st.button("🗑️ Clear History", on_click=clear_history)
    else:
        st.info("No scans yet. Upload a leaf image above to get started!")


# ============================================================
# PAGE 2 — VIEW RESULTS (ANALYTICS)
# ============================================================
elif app_mode == "📊 View Results":
    st.markdown("""
        <h1 style='text-align: center;'>📊 View Results</h1>
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
        st.markdown("""
        <div class="metric-card">
            <div class="icon">🖼️</div>
            <div class="label">Total Images</div>
            <div class="value">54,306</div>
            <div class="note note-blue">Training data</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="icon">🏷️</div>
            <div class="label">Disease Types</div>
            <div class="value">38</div>
            <div class="note note-blue">Recognized</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="icon">📤</div>
            <div class="label">Uploads</div>
            <div class="value">{st.session_state.upload_count}</div>
            <div class="note note-blue">This session</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="icon">🎯</div>
            <div class="label">Accuracy</div>
            <div class="value">95.28%</div>
            <div class="note note-green">Verified</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="icon">🚨</div>
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
        st.markdown("### 🌾 Images per Crop")

        df_crops = pd.DataFrame({
            'Crop': ['🍅 Tomato', '🥔 Potato', '🌶️ Pepper'],
            'Images': [18160, 2152, 2475]
        })

        fig_bar = px.bar(
            df_crops, x='Crop', y='Images',
            color='Crop',
            color_discrete_map={
                '🍅 Tomato': '#ff6b6b',
                '🥔 Potato': '#f0c040',
                '🌶️ Pepper': '#4cd964'
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
        st.markdown("### 🩺 Crop Health Overview")

        df_health = pd.DataFrame({
            'Status': ['Healthy', 'Diseased'],
            'Count': [4827, 17960]
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
            <h2 style='text-align: center;'>📈 Your Session</h2>
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
                <div class="icon">🌱</div>
                <div class="label">Plants Scanned</div>
                <div class="value">{df_session['Plant Name'].nunique()}</div>
                <div class="note note-blue">Unique crops</div>
            </div>""", unsafe_allow_html=True)

        with s2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">🔬</div>
                <div class="label">Diseases Found</div>
                <div class="value">{df_session['Disease'].nunique()}</div>
                <div class="note note-blue">Unique types</div>
            </div>""", unsafe_allow_html=True)

        with s3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">✅</div>
                <div class="label">Healthy</div>
                <div class="value">{healthy_count}</div>
                <div class="note note-green">Looking good</div>
            </div>""", unsafe_allow_html=True)

        with s4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">⚠️</div>
                <div class="label">Diseased</div>
                <div class="value">{diseased_session}</div>
                <div class="note note-red">Needs attention</div>
            </div>""", unsafe_allow_html=True)

        # Session crop breakdown chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌿 Scans by Crop")

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
            🌿 Crop Disease Detector
        </p>
        <p style='color: #5a6a9a; font-size: 0.78rem; margin: 0.3rem 0 0 0;'>
            AI Powered · Plant Image Data
        </p>
        <p style='color: #5a6a9a; font-size: 0.75rem; margin: 0.5rem 0 0 0;'>
            © 2026 All Rights Reserved
        </p>
        <p style='color: #4a90e2; font-size: 0.75rem; margin: 0.3rem 0 0 0;'>
            Developed with ❤️ by Awowole Hammad Olamilekan · FTP/CSC/25/0118830
        </p>
    </div>
    """, unsafe_allow_html=True)
