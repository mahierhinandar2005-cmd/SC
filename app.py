import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Battery Health Predictor", page_icon="🔋", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .soh-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
    }
    .soh-critical {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .info-btn {
        background: none;
        border: none;
        color: #667eea;
        font-size: 0.9rem;
        cursor: pointer;
        margin-left: 5px;
    }
    .info-btn:hover {
        color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔋 Battery Health Predictor</h1>
    <p>Electric Vehicle State of Health (SOH) Estimation using ANN</p>
    <p style="font-size: 0.8rem;">Project SC 2026 | ANN-based Prediction</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/battery-charging.png", width=80)
    st.markdown("### About")
    st.info("""
    **Metode:** ANN (MLPRegressor)
    
    **Dataset:** CNR Italy EIS Dataset (Cell 08)
    
    **Fitur:** Aging cycle, SOC, R_int, OCV
    
    **Output:** State of Health (SOH) & Recommendation
    """)
    st.markdown("---")
    st.caption("© 2026 | Built for Academic Project")

@st.cache_resource
def load_model():
    model = joblib.load('model_soh.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.subheader("📝 Battery Parameters Input")

# Fungsi untuk bikin info popup
def info_button(param_name, description):
    if st.button("ℹ️", key=param_name, help=description):
        st.toast(f"📌 {description}", icon="ℹ️")

col1, col2 = st.columns(2)

with col1:
    # Aging Cycle dengan tombol info
    cycle_label, cycle_info = st.columns([4, 1])
    with cycle_label:
        st.markdown("**🔄 Aging Cycle**")
    with cycle_info:
        st.button("ℹ️", key="info_cycle", help="Jumlah siklus charge-discharge yang sudah dilalui baterai. Rentang: 0 - 2000 cycle.")
    
    cycle = st.number_input("", min_value=0, max_value=2000, value=100, step=10, key="cycle_input", label_visibility="collapsed")
    
    # SOC dengan tombol info
    soc_label, soc_info = st.columns([4, 1])
    with soc_label:
        st.markdown("**🔋 SOC (%)**")
    with soc_info:
        st.button("ℹ️", key="info_soc", help="State of Charge - Level persentase pengisian baterai saat ini. Normal: 20% - 80%.")
    
    soc = st.number_input("", min_value=0, max_value=100, value=80, step=5, key="soc_input", label_visibility="collapsed")

with col2:
    # R_int dengan tombol info
    rint_label, rint_info = st.columns([4, 1])
    with rint_label:
        st.markdown("**⚡ R_int (%)**")
    with rint_info:
        st.button("ℹ️", key="info_rint", help="Internal Resistance - Resistansi internal baterai. Naik >100% menandakan degradasi.")
    
    r_int = st.number_input("", min_value=0, max_value=200, value=10, step=5, key="rint_input", label_visibility="collapsed")
    
    # OCV dengan tombol info
    ocv_label, ocv_info = st.columns([4, 1])
    with ocv_label:
        st.markdown("**🔌 OCV (V)**")
    with ocv_info:
        st.button("ℹ️", key="info_ocv", help="Open Circuit Voltage - Tegangan baterai saat tidak dibebani. Sehat: >3.8V.")
    
    ocv = st.number_input("", min_value=3.0, max_value=4.5, value=4.1, step=0.05, key="ocv_input", label_visibility="collapsed")

if st.button("🔮 Predict Battery Health", type="primary", use_container_width=True):
    input_data = np.array([[cycle, soc, r_int, ocv]])
    input_scaled = scaler_X.transform(input_data)
    pred_scaled = model.predict(input_scaled)
    soh = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    if soh >= 90:
        status = "✅ SEHAT"
        status_desc = "Battery in excellent condition"
        recommendation = "Continue normal usage. Next service in 6 months."
        color_class = "soh-card"
    elif soh >= 70:
        status = "⚠️ WASPADA"
        status_desc = "Battery showing degradation"
        recommendation = "Schedule battery inspection and balancing service soon."
        color_class = "soh-card"
    else:
        status = "🔴 KRITIS"
        status_desc = "Battery health critical"
        recommendation = "Battery replacement strongly recommended."
        color_class = "soh-critical"
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="{color_class}">
            <h3>State of Health</h3>
            <h1 style="font-size: 3rem;">{soh:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="soh-card">
            <h3>Status</h3>
            <h2>{status}</h2>
            <p>{status_desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="soh-card">
            <h3>Recommendation</h3>
            <p>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={"text": "SOH Meter"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#667eea"},
            "steps": [
                {"range": [0, 70], "color": "#eb3349"},
                {"range": [70, 90], "color": "#f2994a"},
                {"range": [90, 100], "color": "#11998e"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": soh
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Dataset: CNR Italy EIS Dataset | Model: ANN (MLPRegressor) | Project SC 2026</p>
</div>
""", unsafe_allow_html=True)
