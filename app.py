"""
app.py – GAVA AI: J.A.R.V.I.S. HUD Command Center
=================================================
Theme: Iron Man HUD (Cyan #00F3FF, Amber #FFB000, Deep Navy/Black)
Features:
  - Base64 injected HTML/CSS for advanced layouts
  - Laser connector lines from action frame to zoomed face
  - Lively typewriter animation for biometric details
  - Glowing corner-bracket targeting reticles
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import tempfile
from pathlib import Path
import streamlit as st

# Pipeline imports
from run_pipeline import run_pipeline
from config import EVIDENCE_DIR, REPORTS_DIR

# ── Helper: Image to Base64 for HTML Rendering ────────────────────────────────
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ── 1. Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GAVA AI // TACTICAL HUD",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── 2. J.A.R.V.I.S. HUD CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    /* Global Dark Background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #020611 !important;
        background-image: 
            linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        color: #00f3ff !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(2, 6, 17, 0.95) !important;
        border-right: 1px solid rgba(0, 243, 255, 0.3) !important;
    }

    /* Titles */
    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #00f3ff;
        text-shadow: 0 0 10px #00f3ff, 0 0 20px #00f3ff;
        letter-spacing: 4px;
        margin-bottom: 0px;
        text-transform: uppercase;
    }
    .hud-subtitle {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        color: #ffb000;
        letter-spacing: 2px;
        margin-bottom: 30px;
        text-shadow: 0 0 8px #ffb000;
    }

    /* Action Buttons */
    .stButton>button {
        background: rgba(0, 243, 255, 0.1) !important;
        color: #00f3ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        border: 1px solid #00f3ff !important;
        box-shadow: inset 0 0 10px rgba(0, 243, 255, 0.5), 0 0 15px rgba(0, 243, 255, 0.3) !important;
        border-radius: 0px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: rgba(0, 243, 255, 0.3) !important;
        box-shadow: inset 0 0 20px rgba(0, 243, 255, 0.8), 0 0 30px rgba(0, 243, 255, 0.6) !important;
        text-shadow: 0 0 5px #fff;
    }

    /* ── IRON MAN BIOMETRIC HUD LAYOUT ── */
    .biometric-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 40px;
        gap: 0px;
        width: 100%;
    }

    /* Image Reticle Boxes */
    .target-box {
        position: relative;
        padding: 10px;
        background: rgba(0, 20, 40, 0.6);
        border: 1px solid rgba(0, 243, 255, 0.2);
    }
    .target-box::before, .target-box::after {
        content: '';
        position: absolute;
        width: 20px; height: 20px;
        border-color: #00f3ff;
        border-style: solid;
        transition: all 0.3s;
    }
    .target-box::before {
        top: -2px; left: -2px;
        border-width: 2px 0 0 2px;
    }
    .target-box::after {
        bottom: -2px; right: -2px;
        border-width: 0 2px 2px 0;
    }
    .target-box img {
        display: block;
        max-height: 350px;
        border: 1px solid rgba(0, 243, 255, 0.4);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
    }

    /* Connector Lines */
    .connector-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 120px;
    }
    .laser-line {
        width: 100%;
        height: 2px;
        background: #00f3ff;
        box-shadow: 0 0 10px #00f3ff, 0 0 20px #00f3ff;
        position: relative;
        animation: pulse-laser 2s infinite;
    }
    .laser-line::after {
        content: '';
        position: absolute;
        right: 0;
        top: -4px;
        width: 10px;
        height: 10px;
        background: #fff;
        border-radius: 50%;
        box-shadow: 0 0 10px #fff, 0 0 20px #00f3ff;
    }
    @keyframes pulse-laser {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }

    /* Data Display & Typewriter Effect */
    .data-panel {
        background: rgba(0, 15, 25, 0.8);
        border-left: 3px solid #00f3ff;
        padding: 20px;
        width: 350px;
        box-shadow: inset 20px 0 30px -20px rgba(0, 243, 255, 0.3);
    }
    .typewriter-text {
        overflow: hidden;
        white-space: nowrap;
        margin: 0;
        font-size: 1.1rem;
        color: #00f3ff;
        border-right: 2px solid #00f3ff;
        animation: typing 1.5s steps(40, end) forwards, blink 0.75s step-end infinite;
        width: 0; /* starts hidden */
    }
    .label { color: #888; font-size: 0.9rem; margin-bottom: 2px; margin-top: 10px; }
    
    /* Animation Delays for sequential typing */
    .delay-1 { animation-delay: 0.5s, 0s; }
    .delay-2 { animation-delay: 2.0s, 0s; }
    .delay-3 { animation-delay: 3.5s, 0s; }
    .delay-4 { animation-delay: 5.0s, 0s; }
    
    /* Warning/Alert text */
    .alert-text {
        color: #ff003c;
        text-shadow: 0 0 10px #ff003c;
        font-weight: bold;
    }

    @keyframes typing {
        from { width: 0; }
        to { width: 100%; border-right: transparent; }
    }
    @keyframes blink {
        from, to { border-color: transparent; }
        50% { border-color: #00f3ff; }
    }
</style>
""", unsafe_allow_html=True)

# ── 3. Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hud-title">SYSTEM: G.A.V.A AI.</div>', unsafe_allow_html=True)
st.markdown('<div class="hud-subtitle">TACTICAL SURVEILLANCE & BIOMETRIC RECOGNITION NETWORK</div>', unsafe_allow_html=True)

# ── 4. Input & Execution ──────────────────────────────────────────────────────
col_input, col_run = st.columns([1, 1])

with col_input:
    input_mode = st.radio("SELECT FEED:", ["UPLOAD STREAM", "LOAD SAMPLE"], horizontal=True)
    selected_video_path = None
    if input_mode == "UPLOAD STREAM":
        uploaded_file = st.file_uploader("INITIALIZE VIDEO FEED", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            selected_video_path = tfile.name
    else:
        sample_dir = Path("sample_data")
        if sample_dir.exists():
            video_files = [f.name for f in sample_dir.glob("*.mp4")]
            if video_files:
                chosen = st.selectbox("AVAILABLE SECTORS:", video_files)
                selected_video_path = str(sample_dir / chosen)

with col_run:
    st.markdown("<br><br>", unsafe_allow_html=True)
    run_btn = st.button("INITIATE TACTICAL SCAN", disabled=(selected_video_path is None))

if run_btn and selected_video_path:
    with st.status("ESTABLISHING UPLINK...", expanded=True) as status:
        st.write(">> ISOLATING KINEMATIC VECTORS...")
        st.write(">> RUNNING INCEPTION-RESNET-V1 BIOMETRIC SWEEP...")
        st.write(">> ANALYZING MATERIAL REFLECTANCE SPECTRA...")
        st.write(">> COMPILING TACTICAL DOSSIER...")

        try:
            incident_payload, pdf_path = run_pipeline(selected_video_path)
            status.update(label=">> TARGET ACQUIRED. AUDIT COMPILED.", state="complete", expanded=False)
            st.session_state["pipeline_result"] = incident_payload
            st.session_state["pdf_path"] = pdf_path
        except Exception as e:
            status.update(label=">> SYSTEM FAILURE", state="error")
            st.error(f"ERR: {str(e)}")

# ── 5. IRON MAN HUD RENDER ────────────────────────────────────────────────────
if "pipeline_result" in st.session_state and st.session_state["pipeline_result"]:
    res = st.session_state["pipeline_result"]
    pdf_file_path = st.session_state.get("pdf_path")

    # Name cleaning
    offender_name = res["track_info"]["student_name"].replace("MITS_", "").replace("MITS-", "").replace("MITS", "").strip()
    if not offender_name: offender_name = "UNKNOWN_SUBJECT"
    
    match_conf = float(res["track_info"]["match_confidence"]) * 100
    item_class = res["waste_result"]["predicted_class"].upper()
    fine_amt = res["fine_amount"]

    # Convert images to Base64
    action_b64 = get_base64_image(res["littering_event"]["evidence_path"])
    face_b64 = get_base64_image(res["person_crop_path"])

    if action_b64 and face_b64:
        # Build the HTML structure
        hud_html = f"""
        <div class="biometric-container">
            
            <!-- FIGURE A: Full Action Frame -->
            <div class="target-box">
                <div style="position: absolute; top: 15px; left: 15px; color: #ffb000; font-size: 0.8rem; background: rgba(0,0,0,0.5); padding: 2px 6px;">CAM_01 // TRACKING</div>
                <img src="data:image/jpeg;base64,{action_b64}" alt="Action Frame">
            </div>

            <!-- Connector Laser Lines -->
            <div class="connector-wrapper">
                <div style="color: #00f3ff; font-size: 0.7rem; margin-bottom: 5px;">EXTRACTING</div>
                <div class="laser-line"></div>
                <div style="color: #00f3ff; font-size: 0.7rem; margin-top: 5px;">BIOMETRICS</div>
            </div>

            <!-- FIGURE B: Zoomed Face -->
            <div class="target-box">
                <div style="position: absolute; top: 15px; left: 15px; color: #00f3ff; font-size: 0.8rem; background: rgba(0,0,0,0.5); padding: 2px 6px;">FACE_MATCH // {match_conf:.1f}%</div>
                <img src="data:image/jpeg;base64,{face_b64}" alt="Zoomed Face" style="height: 250px; width: 250px; object-fit: cover;">
            </div>

            <!-- Dynamic Typed Data HUD -->
            <div class="data-panel">
                <div class="label">SUBJECT_ID //</div>
                <p class="typewriter-text delay-1">{offender_name}</p>
                
                <div class="label">MATCH_PROBABILITY //</div>
                <p class="typewriter-text delay-2">{match_conf:.2f}% (VERIFIED)</p>
                
                <div class="label">DISCARDED_OBJECT //</div>
                <p class="typewriter-text delay-3 alert-text">[{item_class}] DETECTED</p>
                
                <div class="label">ENFORCEMENT_ACTION //</div>
                <p class="typewriter-text delay-4" style="color: #ffb000;">PENALTY: INR {fine_amt}</p>
            </div>

        </div>
        """
        st.markdown(hud_html, unsafe_allow_html=True)
    else:
        st.error(">> ERR: IMAGE BUFFER NOT FOUND.")

    # PDF Download Button
    st.markdown("<br><br>", unsafe_allow_html=True)
    if pdf_file_path and os.path.exists(pdf_file_path):
        with open(pdf_file_path, "rb") as f:
            pdf_data = f.read()

        st.download_button(
            label=">> DOWNLOAD TACTICAL DOSSIER [PDF] <<",
            data=pdf_data,
            file_name=os.path.basename(pdf_file_path),
            mime="application/pdf",
            use_container_width=True
        )