"""Compact Fig. 14-inspired predictor for the exported N_test models."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model_loader import MODEL_ORDER, load_bundle
from prediction import predict_all
from sensitivity import sensitivity_data
from uq import predict_with_uq, test_uq_bundle


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "gui_export_bundle.json"
if not BUNDLE_PATH.exists():
    BUNDLE_PATH = APP_DIR / "gui_export_bundle (1).json"
CORE_FEATURES = ["D", "t", "L", "fy"]
ADDITIONAL_FEATURES = ["Ea", "fc", "et", "eb"]

st.set_page_config(page_title="N_test Predictor", page_icon="📐", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root{--bg:#f4f9fd;--panel:#ffffff;--panel2:#e8f3fb;--blue:#146fae;--orange:#0b4776;--text:#12344f;--muted:#5d7890;--line:#c5ddea}
    .stApp{background:linear-gradient(135deg,#eef7fc 0,#ffffff 58%,#e6f2fa 100%);color:var(--text);font-family:'DM Sans',sans-serif}
    .block-container{max-width:1180px;padding:2.5rem 3rem 4rem}
    h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:0;color:var(--orange)}
    .title{text-align:center;background:linear-gradient(110deg,#0b4776,#146fae);border-radius:12px;padding:1.8rem 1rem;margin-bottom:1.4rem;box-shadow:0 10px 24px #146fae22}
    .title h1{font-size:2.7rem;margin:.2rem 0 .35rem;color:#ffffff}.title p{color:#dceefa;font-size:1.05rem;margin:0}
    .stTabs [data-baseweb='tab-list']{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.25rem;gap:.25rem}
    .stTabs [data-baseweb='tab']{color:var(--muted);font-weight:700;padding:.65rem 1.2rem}.stTabs [aria-selected='true']{background:var(--panel2);color:var(--blue);border-radius:6px}
    .section{font:700 1.1rem 'Space Grotesk',sans-serif;color:var(--orange);border-left:5px solid var(--blue);border-bottom:1px solid var(--line);padding:.45rem .7rem;margin:1.2rem 0 .8rem;background:#eaf5fb}
    .stSlider{padding-top:0}.stSlider [data-baseweb='slider'] div{background-color:var(--blue)}
    .stButton>button{background:var(--blue);color:#ffffff;border:0;border-radius:6px;font:700 1rem 'Space Grotesk',sans-serif;min-height:2.9rem;box-shadow:0 6px 14px #146fae33}
    .stButton>button:hover{background:#0b4776;color:#ffffff}.calculate-wrap{text-align:center;margin:1.7rem auto 1.3rem;max-width:340px}
    .result{background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:7px;padding:1rem;text-align:center;min-height:245px;box-shadow:0 8px 20px #146fae18}
    .result .model{color:var(--blue);font:700 .82rem 'Space Grotesk',sans-serif;letter-spacing:.12em;text-transform:uppercase}.result .label{color:var(--muted);margin:.55rem 0 .2rem}.result .number{font:700 1.65rem 'Space Grotesk',sans-serif;color:var(--orange)}
    .interval{border-radius:5px;padding:.45rem .35rem;margin-top:.7rem;font-size:.85rem;color:var(--text)}.interval span{display:block;color:var(--muted);font-size:.72rem;margin-bottom:.15rem}.pi90{background:#dceefa}.pi95{background:#edf3f7}.width-note{color:var(--muted);font-size:.72rem;margin-top:.65rem}
    [data-testid='stNumberInput'] input{border:1px solid var(--line);border-radius:5px;background:#ffffff;color:var(--text)}
    [data-testid='stAlert']{border-radius:7px}.stMarkdown,.stCaption{color:var(--muted)}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_bundle(path: str):
    return load_bundle(path)


def sync_input(source_key: str, target_key: str) -> None:
    st.session_state[target_key] = st.session_state[source_key]


try:
    bundle = get_bundle(str(BUNDLE_PATH))
    test_uq_bundle(bundle)
except Exception as exc:
    st.error("Unable to load or validate the model bundle. Place the exported JSON beside app.py.")
    st.exception(exc)
    st.stop()


st.markdown(
    f'<div class="title"><h1>{bundle["target_name"]} Predictor</h1><p>Machine-learning based prediction tool | Target: {bundle["target_name"]} ({bundle.get("target_unit", "")})</p></div>',
    unsafe_allow_html=True,
)

tab_predictor, tab_information = st.tabs(["Predictor", "Model Information"])

with tab_predictor:
    st.markdown('<div class="section">Input Parameters</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    values = {}
    for column, features, heading in ((left, CORE_FEATURES, "Core Parameters"), (right, ADDITIONAL_FEATURES, "Additional Parameters")):
        with column:
            st.markdown(f'<div class="section">{heading}</div>', unsafe_allow_html=True)
            for feature in features:
                limits = bundle["feature_ranges"][feature]
                low, high, default = float(limits["min"]), float(limits["max"]), float(limits["median"])
                manual_key = f"manual_{feature}"
                slider_key = f"slider_{feature}"
                if manual_key not in st.session_state:
                    st.session_state[manual_key] = float(np.clip(default, low, high))
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = st.session_state[manual_key]
                st.number_input(
                    f"{feature} value",
                    min_value=low,
                    max_value=high,
                    key=manual_key,
                    format="%.6g",
                    on_change=sync_input,
                    args=(manual_key, slider_key),
                )
                st.slider(
                    feature,
                    min_value=low,
                    max_value=high,
                    key=slider_key,
                    on_change=sync_input,
                    args=(slider_key, manual_key),
                    help=f"Validated range: {low:g} to {high:g}",
                )
                values[feature] = float(st.session_state[manual_key])

    with st.container():
        st.markdown('<div class="calculate-wrap">', unsafe_allow_html=True)
        calculate = st.button("Calculate N_test", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if calculate:
        rows = np.asarray([[values[feature] for feature in bundle["feature_order"]]], dtype=float)
        st.session_state["predictions"] = {model: float(prediction[0]) for model, prediction in predict_all(bundle, rows).items()}
        st.session_state["uq_results"] = {model: predict_with_uq(bundle, model, rows) for model in MODEL_ORDER}
        st.session_state["calculated_values"] = values.copy()

    st.markdown('<div class="section">Predicted Results</div>', unsafe_allow_html=True)
    result_columns = st.columns(3, gap="medium")
    predictions = st.session_state.get("predictions", {})
    uq_results = st.session_state.get("uq_results", {})
    for column, model in zip(result_columns, MODEL_ORDER):
        with column:
            if model in predictions and uq_results.get(model) is not None:
                result = uq_results[model]
                display_value = f'{predictions[model]:,.2f} {bundle.get("target_unit", "")}'
                pi90 = f'[{result["pi90_lower"][0]:,.2f}, {result["pi90_upper"][0]:,.2f}] {bundle.get("target_unit", "")}'
                pi95 = f'[{result["pi95_lower"][0]:,.2f}, {result["pi95_upper"][0]:,.2f}] {bundle.get("target_unit", "")}'
                card = f'<div class="result"><div class="model">{model}</div><div class="label">Predicted N_test</div><div class="number">{display_value}</div><div class="interval pi90"><span>90% Prediction Interval</span>{pi90}</div><div class="interval pi95"><span>95% Prediction Interval</span>{pi95}</div><div class="width-note">Width 90%: {result["width90"][0]:,.2f} | Width 95%: {result["width95"][0]:,.2f}</div></div>'
            else:
                card = f'<div class="result"><div class="model">{model}</div><div class="label">UQ unavailable</div><div class="number">--</div></div>'
            st.markdown(card, unsafe_allow_html=True)

    if predictions:
        uq_figure = go.Figure()
        model_positions = np.arange(len(MODEL_ORDER))
        for model, position in zip(MODEL_ORDER, model_positions):
            result = uq_results[model]
            is_first_model = bool(position == 0)
            uq_figure.add_trace(go.Scatter(x=[position, position], y=[result["pi95_lower"][0], result["pi95_upper"][0]], mode="lines", line=dict(color="#8ea8b9", width=12), name="95% PI" if is_first_model else None, showlegend=is_first_model))
            uq_figure.add_trace(go.Scatter(x=[position, position], y=[result["pi90_lower"][0], result["pi90_upper"][0]], mode="lines", line=dict(color="#2587c4", width=8), name="90% PI" if is_first_model else None, showlegend=is_first_model))
            uq_figure.add_trace(go.Scatter(x=[position], y=[predictions[model]], mode="markers", marker=dict(color="#f0a35b", size=10), name="Prediction" if is_first_model else None, showlegend=is_first_model))
        uq_figure.update_layout(title="Prediction and uncertainty by model", template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#102131", xaxis=dict(tickmode="array", tickvals=list(model_positions), ticktext=MODEL_ORDER, title="Model"), yaxis_title=f"N_test ({bundle.get('target_unit', '')})", margin=dict(l=55,r=20,t=60,b=45))
        st.plotly_chart(uq_figure, use_container_width=True)

    st.markdown('<div class="section">Variable Impact Analysis</div>', unsafe_allow_html=True)
    open_analysis = st.checkbox("Open for Variable Impact Analysis")
    if open_analysis:
        selected_feature = st.selectbox("Select Variable to Plot", bundle["feature_order"], key="impact_feature")
        selected_model = st.selectbox("Model", MODEL_ORDER, key="impact_model")
        generate_plot = st.button("Generate Plot", key="generate_impact_plot")
        if generate_plot:
            baseline = st.session_state.get("calculated_values", values)
            data = sensitivity_data(bundle, selected_model, selected_feature, baseline, points=80)
            figure = go.Figure()
            if "pi95_lower" in data:
                figure.add_trace(go.Scatter(x=np.r_[data["x"], data["x"][::-1]], y=np.r_[data["pi95_upper"], data["pi95_lower"][::-1]], fill="toself", line=dict(width=0), fillcolor="#8ea8b9", opacity=.35, name="95% PI"))
                figure.add_trace(go.Scatter(x=np.r_[data["x"], data["x"][::-1]], y=np.r_[data["pi90_upper"], data["pi90_lower"][::-1]], fill="toself", line=dict(width=0), fillcolor="#2587c4", opacity=.3, name="90% PI"))
            figure.add_trace(go.Scatter(x=data["x"], y=data["prediction"], mode="lines+markers", name=f"{selected_model} prediction", line=dict(color="#2587c4", width=3), marker=dict(size=4)))
            figure.add_trace(go.Scatter(x=[baseline[selected_feature]], y=[float(predict_all(bundle, np.asarray([[baseline[feature] for feature in bundle["feature_order"]]], dtype=float))[selected_model][0])], mode="markers", name="Current setting", marker=dict(symbol="star", size=15, color="#ed7650", line=dict(color="#fff", width=1))))
            figure.update_layout(title=f"Variable impact: {selected_feature}", xaxis_title=selected_feature, yaxis_title=f"Predicted {bundle['target_name']} ({bundle.get('target_unit', '')})", template="plotly_dark", height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#102131", font=dict(color="#f5f8fb"), hovermode="x unified", margin=dict(l=55, r=25, t=65, b=55))
            st.plotly_chart(figure, use_container_width=True)
            st.caption("The selected variable is scanned from its validated minimum to maximum. All other variables remain fixed at the current slider setting.")

with tab_information:
    st.markdown('<div class="section">Model Information</div>', unsafe_allow_html=True)
    st.write(f"Target: {bundle['target_name']} ({bundle.get('target_unit', '')})")
    st.write(f"Training samples: {bundle.get('n_train', 'not exported')} | Test samples: {bundle.get('n_test', 'not exported')}")
    st.write("Models: RF, ETR, DTR")
    st.write("Input order: " + ", ".join(bundle["feature_order"]))
    st.caption("No model training or uncertainty calculation is performed in this first interface version.")
