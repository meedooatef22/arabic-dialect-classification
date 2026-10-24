import streamlit as st
import torch
import os
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Arabic Dialect Classification Platform",
    page_icon="🌍",
    layout="centered"
)

if 'first_load' not in st.session_state:
    st.session_state.first_load = True
    st.balloons()

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span, button[data-baseweb="tab"], .stTextArea label {
        color: #ffffff !important;
    }
    .footer-text {
        text-align: center !important;
        color: #94a3b8;
        font-size: 13px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSelectbox div, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    .main-title {
        text-align: center;
        color: #f8fafc;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0px;
        text-shadow: 0 2px 10px rgba(59, 130, 246, 0.5);
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 14px;
        padding: 12px;
        border: none;
        box-shadow: 0 6px 15px rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
    }
    .result-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-top: 20px;
        border: 2px solid #6366f1;
    }
    .result-box h3, .result-box p, .result-box b, .result-box span {
        color: #1e293b !important;
    }
    .highlight-dialect {
        color: #2563eb !important;
        font-size: 21px;
        font-weight: 800;
    }
    .highlight-conf {
        color: #16a34a !important;
        font-size: 19px;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌍 Smart Arabic Dialect Classification Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">State-of-the-Art Transformer Models for Fine-Grained Arabic Dialect Identification (MARBERTv2 & AraBERTv02)</p>', unsafe_allow_html=True)

# 1. Exact Text Cleaning Function (Matches Training Pipeline 100%)
def clean_arabic_text(text):
    text = str(text)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652\u0640]', '', text) # Remove diacritics & tatweel
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text) # Keep Arabic letters only
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. Dialect Display Names Mapping
dialect_display = {
    'Beirut': 'Beirut (Levantine) 🇱🇧',
    'Cairo': 'Cairo (Egyptian) 🇪🇬',
    'Doha': 'Doha (Gulf) 🇶🇦',
    'Gulf': 'Doha (Gulf) 🇶🇦',
    'MSA': 'Modern Standard Arabic (MSA) 📖',
    'Tunis': 'Tunis (Maghrebi) 🇹🇳'
}

# Standard sorted labels used in MARBERTv2 & Baseline
STANDARD_LABELS = ['Beirut', 'Cairo', 'Doha', 'MSA', 'Tunis']

# Correct Paths inside the Project Folder on Google Drive
PROJECT_ROOT = "/content/drive/MyDrive/arabic-dialect-classification"
MARBERT_PATH = os.path.join(PROJECT_ROOT, "models/marbert_best_checkpoint")

# Check multiple possible paths for AraBERT
ARABERT_PATHS = [
    os.path.join(PROJECT_ROOT, "models/Arabert_dialect_final_model"),
    "/content/drive/MyDrive/Arabert_dialect_final_model"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_specific_model(model_type):
    if model_type == "marbert":
        if os.path.exists(MARBERT_PATH):
            tokenizer = AutoTokenizer.from_pretrained(MARBERT_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MARBERT_PATH).to(device)
            return tokenizer, model, STANDARD_LABELS, True
        else:
            tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")
            model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERTv2", num_labels=5).to(device)
            return tokenizer, model, STANDARD_LABELS, False

    elif model_type == "arabert":
        for path in ARABERT_PATHS:
            if os.path.exists(path):
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
                train_csv_path = os.path.join(PROJECT_ROOT, "data/processed/train.csv")
                if os.path.exists(train_csv_path):
                    df_tr = pd.read_csv(train_csv_path)
                    ara_labels = list(df_tr['label'].dropna().unique())
                else:
                    ara_labels = STANDARD_LABELS
                return tokenizer, model, ara_labels, True
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
        model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02", num_labels=5).to(device)
        return tokenizer, model, STANDARD_LABELS, False

def predict_dialect(text, tokenizer, model, label_list):
    cleaned = clean_arabic_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))

    raw_label = label_list[pred_id] if pred_id < len(label_list) else "Unknown"
    best_dialect = dialect_display.get(raw_label, raw_label)
    confidence = float(probs[pred_id]) * 100

    all_scores = {dialect_display.get(label_list[i], label_list[i]): float(probs[i]) * 100 for i in range(len(label_list))}
    return best_dialect, confidence, all_scores, cleaned

# Sidebar Configuration
st.sidebar.header("⚙️ Platform Configuration")
model_choice = st.sidebar.selectbox(
    "Select Primary Inference Model:",
    ("MARBERTv2 (Champion - 92.25% Acc)", "AraBERTv02 (91.03% Acc)")
)

is_arabert = "AraBERT" in model_choice
active_tokenizer, active_model, current_labels, is_loaded_ok = load_specific_model("arabert" if is_arabert else "marbert")

if is_loaded_ok:
    st.sidebar.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 14px; border-radius: 12px; color: #ffffff; font-size: 14px; font-weight: bold; margin-top: 15px; text-align: center;">
            ✅ Fine-Tuned Weights Loaded ({'GPU' if device.type=='cuda' else 'CPU'}) 🚀
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.error("⚠️ Warning: Model checkpoint not found on Google Drive! Verify folder paths.")

tab1, tab2, tab3 = st.tabs(["🎯 Single Sentence Analysis", "⚖️ Head-to-Head Comparison", "📁 Batch CSV Inference"])

with tab1:
    user_input = st.text_area("✍️ Enter Arabic text to classify:", placeholder="Example: شورايك بعشره دولار للتلاته / عامل إيه يا صاحبي / يعيشك خويا...", height=120)

    if st.button("🚀 Analyze & Predict Dialect"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter an Arabic sentence first!")
        else:
            with st.spinner('⏳ Running preprocessing and neural inference...'):
                best_dialect, confidence, all_scores, cleaned_txt = predict_dialect(
                    user_input, active_tokenizer, active_model, current_labels
                )

            neutral_note = ""
            if confidence < 45.0:
                neutral_note = '<p style="color: #d97706 !important; font-size: 15px;">⚠️ <b>Linguistic Note:</b> Low confidence score indicates a short or dialect-neutral phrase shared across multiple regions.</p>'

            st.markdown(
                f"""
                <div class="result-box">
                    <h3 style="margin-top: 0; color: #4f46e5;">📊 Classification Results:</h3>
                    <p style="font-size: 19px;">🎯 <b>Predicted Dialect:</b> <span class="highlight-dialect">{best_dialect}</span></p>
                    <p style="font-size: 18px;">📈 <b>Confidence Score:</b> <span class="highlight-conf">{confidence:.2f}%</span></p>
                    <p style="font-size: 14px; color: #64748b !important;">🧹 <b>Normalized Text:</b> {cleaned_txt}</p>
                    {neutral_note}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### 📊 Confidence Breakdown Across All 5 Dialects:")
            scores_df = pd.DataFrame(list(all_scores.items()), columns=['Dialect', 'Confidence (%)']).sort_values(by='Confidence (%)', ascending=False)
            for _, row in scores_df.iterrows():
                st.write(f"**{row['Dialect']}** — `{row['Confidence (%)']:.2f}%`")
                st.progress(min(max(int(row['Confidence (%)']), 1), 100))

with tab2:
    st.markdown("### ⚖️ Real-Time Model Comparison (MARBERTv2 vs. AraBERTv02)")
    comp_input = st.text_area("✍️ Enter an Arabic sentence to test on both models simultaneously:", placeholder="Example: شورايك بعشره دولار للتلاته", height=100, key="comp_box")

    if st.button("🔍 Compare Models Now"):
        if comp_input.strip() == "":
            st.warning("⚠️ Please enter a sentence for comparison!")
        else:
            col_a, col_b = st.columns(2)

            tok_m, mod_m, lbl_m, _ = load_specific_model("marbert")
            res_m, conf_m, _, _ = predict_dialect(comp_input, tok_m, mod_m, lbl_m)

            tok_a, mod_a, lbl_a, _ = load_specific_model("arabert")
            res_a, conf_a, _, _ = predict_dialect(comp_input, tok_a, mod_a, lbl_a)

            with col_a:
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 20px; border-radius: 15px; border: 2px solid #ec4899;">
                        <h4 style="color: #f472b6; margin-top: 0;">🏆 MARBERTv2 (92.25% Test Acc)</h4>
                        <p style="color: white;">🎯 Dialect: <b>{res_m}</b></p>
                        <p style="color: white;">📈 Confidence: <b>{conf_m:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_b:
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 20px; border-radius: 15px; border: 2px solid #3b82f6;">
                        <h4 style="color: #60a5fa; margin-top: 0;">🤖 AraBERTv02 (91.03% Val Acc)</h4>
                        <p style="color: white;">🎯 Dialect: <b>{res_a}</b></p>
                        <p style="color: white;">📈 Confidence: <b>{conf_a:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

with tab3:
    st.markdown("### 📁 Batch CSV File Analysis")
    st.markdown("Upload a CSV file containing a `text` column to classify multiple sentences simultaneously using **MARBERTv2**.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            st.success(f"✅ File uploaded successfully ({len(df)} sentences found).")
            if st.button("⚡ Run Batch Classification"):
                tok_m, mod_m, lbl_m, _ = load_specific_model("marbert")
                predictions, confidences = [], []

                with st.spinner('⏳ Classifying batch sentences...'):
                    for text in df['text']:
                        res_m, conf_m, _, _ = predict_dialect(str(text), tok_m, mod_m, lbl_m)
                        predictions.append(res_m)
                        confidences.append(f"{conf_m:.2f}%")

                df['Predicted_Dialect'] = predictions
                df['Confidence_Score'] = confidences
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name='dialect_analysis_results.csv',
                    mime='text/csv',
                )
        else:
            st.error("⚠️ The uploaded CSV file must contain a column named `text`!")

st.markdown("<br><hr style='border-color: #334155;'><p class='footer-text'>Arabic Dialect Classification Graduation Project — Powered by MARBERTv2 & Streamlit 🚀</p>", unsafe_allow_html=True)
