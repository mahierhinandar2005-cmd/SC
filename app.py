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
    .warning-card {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .danger-card {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-card {
        background: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
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

# Informasi singkat di atas form
with st.expander("ℹ️ Panduan Singkat Parameter", expanded=False):
    st.markdown("""
    | Parameter | Penjelasan | Nilai Normal | Nilai Berbahaya |
    |:---|:---|:---|:---|
    | **Aging Cycle** | Jumlah siklus charge-discharge | 0 - 200 | > 600 |
    | **SOC (%)** | Level pengisian baterai | 20% - 80% | <20% atau >80% |
    | **R_int (%)** | Internal resistance | 90% - 110% | > 150% |
    | **OCV (V)** | Open circuit voltage | 3.8 - 4.2 V | < 3.6 V |
    """)

col1, col2 = st.columns(2)

with col1:
    cycle = st.number_input("🔄 Aging Cycle", min_value=0, max_value=2000, value=100, step=10,
                            help="Jumlah siklus charge-discharge. Semakin tinggi, baterai semakin aus.")
    
    soc = st.number_input("🔋 SOC (%)", min_value=0, max_value=100, value=80, step=5,
                          help="State of Charge - Level pengisian baterai. Idealnya 20-80%.")

with col2:
    r_int = st.number_input("⚡ R_int (%)", min_value=0, max_value=200, value=100, step=5,
                            help="Internal resistance - 100% = normal. Diatas 150% indikasi degradasi.")
    
    ocv = st.number_input("🔌 OCV (V)", min_value=3.0, max_value=4.5, value=4.1, step=0.05,
                          help="Open circuit voltage - Tegangan saat diam. Baterai sehat di atas 3.8V.")

# Fungsi untuk mengecek parameter yang bermasalah
def check_parameter_issues(cycle, soc, r_int, ocv):
    issues = []
    
    # Cek Cycle
    if cycle > 600:
        issues.append({
            "param": "Aging Cycle",
            "value": cycle,
            "normal": "0 - 200",
            "severity": "danger" if cycle > 800 else "warning",
            "message": f"Cycle sudah mencapai {cycle}. Baterai sudah melewati 600 cycle, mendekati akhir masa pakai."
        })
    elif cycle > 400:
        issues.append({
            "param": "Aging Cycle",
            "value": cycle,
            "normal": "0 - 200",
            "severity": "warning",
            "message": f"Cycle sudah {cycle}. Mulai memasuki fase menua, perhatikan performa baterai."
        })
    
    # Cek SOC
    if soc < 20:
        issues.append({
            "param": "SOC (%)",
            "value": soc,
            "normal": "20% - 80%",
            "severity": "danger",
            "message": f"SOC hanya {soc}%. Terlalu rendah! Bisa menyebabkan deep discharge yang merusak baterai."
        })
    elif soc > 80:
        issues.append({
            "param": "SOC (%)",
            "value": soc,
            "normal": "20% - 80%",
            "severity": "warning",
            "message": f"SOC {soc}%. Terlalu tinggi. Kebiasaan charge penuh mempercepat degradasi baterai."
        })
    
    # Cek R_int
    if r_int > 150:
        issues.append({
            "param": "R_int (%)",
            "value": r_int,
            "normal": "90% - 110%",
            "severity": "danger",
            "message": f"Internal resistance {r_int}% (naik {r_int-100}% dari normal). Indikasi kuat sel baterai sudah rusak."
        })
    elif r_int > 120:
        issues.append({
            "param": "R_int (%)",
            "value": r_int,
            "normal": "90% - 110%",
            "severity": "warning",
            "message": f"Internal resistance {r_int}% (naik {r_int-100}%). Mulai ada degradasi, segera lakukan inspeksi."
        })
    
    # Cek OCV
    if ocv < 3.6:
        issues.append({
            "param": "OCV (V)",
            "value": ocv,
            "normal": "3.8 - 4.2 V",
            "severity": "danger",
            "message": f"Tegangan OCV hanya {ocv}V. Jauh di bawah normal! Sel baterai kemungkinan sudah rusak parah."
        })
    elif ocv < 3.8:
        issues.append({
            "param": "OCV (V)",
            "value": ocv,
            "normal": "3.8 - 4.2 V",
            "severity": "warning",
            "message": f"Tegangan OCV {ocv}V. Mulai turun, perlu diwaspadai."
        })
    
    return issues

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
    
    # Cek parameter issues (PERINGATAN DINI)
    issues = check_parameter_issues(cycle, soc, r_int, ocv)
    
    if issues:
        st.markdown("---")
        st.markdown("### ⚠️ Early Warning System")
        st.markdown("*Meskipun SOH masih dalam batas normal, ada parameter yang perlu perhatian khusus:*")
        
        for issue in issues:
            if issue["severity"] == "danger":
                st.markdown(f"""
                <div class="danger-card">
                    <strong>🔴 {issue['param']} = {issue['value']} (Normal: {issue['normal']})</strong><br>
                    {issue['message']}<br>
                    <span style="color: #dc3545; font-weight: bold;">→ TINDAKAN: Segera bawa ke bengkel resmi untuk inspeksi menyeluruh!</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <strong>⚠️ {issue['param']} = {issue['value']} (Normal: {issue['normal']})</strong><br>
                    {issue['message']}<br>
                    <span style="color: #856404;">→ TINDAKAN: Monitor secara rutin, lakukan penyesuaian kebiasaan charge.</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown("""
        <div class="info-card">
            <strong>✅ SEMUA PARAMETER DALAM RENTANG NORMAL</strong><br>
            Tidak ada indikasi masalah. Teruskan kebiasaan baik ini untuk menjaga kesehatan baterai.
        </div>
        """, unsafe_allow_html=True)
    
    # Gauge chart
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
    <p>⚠️ Early Warning System: Mendeteksi parameter abnormal sebelum SOH turun drastis</p>
</div>
""", unsafe_allow_html=True)
