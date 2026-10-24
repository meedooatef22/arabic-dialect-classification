import streamlit as st
import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="منصة تصنيف اللهجات العربية الذكية",
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
    
    .main .block-container {
        direction: rtl;
        text-align: right;
    }

    /* إجبار شريط التبويبات بالكامل للتحرك والاصطفاف جهة اليمين تماماً */
    div[data-baseweb="tab-list"] {
        direction: rtl !important;
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
    }
    button[data-baseweb="tab"] {
        direction: rtl !important;
        text-align: right !important;
    }

    button[data-baseweb="tab"] p, 
    button[data-baseweb="tab"] span, 
    button[data-baseweb="tab"], 
    .stTextArea label {
        color: #ffffff !important;
    }

    [data-testid="stHeader"] svg {
        fill: #94a3b8 !important;
    }
    [data-testid="stHeader"] button {
        color: #94a3b8 !important;
    }

    .stTextArea label, .stTextInput label, .stSelectbox label, .stMarkdown h3 {
        direction: rtl !important;
        text-align: right !important;
        float: right !important;
    }
    
    .stTextArea, .stTextInput {
        direction: rtl !important;
        text-align: right !important;
    }

    .footer-text {
        text-align: center !important;
        color: #94a3b8;
        font-size: 13px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
        direction: rtl;
        text-align: right;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSelectbox div, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        text-align: right !important;
    }

    /* تخصيص وضع الـ Light Mode (خلفية فاتحة وكتابة سوداء واضحة للمربعات والصفحة) */
    @media (prefers-color-scheme: light) {
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSelectbox div, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #0f172a !important;
        }
        
        div[data-baseweb="input"] input, 
        div[data-baseweb="textarea"] textarea, 
        input, 
        textarea {
            color: #0f172a !important;
            background-color: #ffffff !important;
            -webkit-text-fill-color: #0f172a !important;
        }
    }

    .main-title {
        text-align: center;
        color: #f8fafc;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0px;
        text-shadow: 0 2px 10px rgba(59, 130, 246, 0.5);
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
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
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(236, 72, 153, 0.6);
    }

    .result-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-top: 20px;
        text-align: right;
        border: 2px solid #6366f1;
    }
    .result-box h3, .result-box p, .result-box b, .result-box span {
        color: #1e293b !important;
    }
    .highlight-dialect {
        color: #2563eb !important;
        font-size: 20px;
        font-weight: 800;
    }
    .highlight-conf {
        color: #16a34a !important;
        font-size: 18px;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">✨ منصة تحليل اللهجات العربية الذكية 🌍</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">أحدث تقنيات الذكاء الاصطناعي لتمييز ومقارنة اللهجات بدقة متناهية</p>', unsafe_allow_html=True)

dialect_name_mapping = {
    'Tunis': 'تونس 🇹🇳',
    'tunis': 'تونس 🇹🇳',
    'Cairo': 'القاهرة (مصر) 🇪🇬',
    'cairo': 'القاهرة (مصر) 🇪🇬',
    'Beirut': 'بيروت (لبنان) 🇱🇧',
    'beirut': 'بيروت (لبنان) 🇱🇧',
    'Gulf': 'الدوحة / الخليج 🇶🇦',
    'gulf': 'الدوحة / الخليج 🇶🇦',
    'MSA': 'اللغة العربية الفصحى AR',
    'msa': 'اللغة العربية الفصحى AR'
}

try:
    train_df_temp = pd.read_csv('/content/drive/MyDrive/train.csv')
    raw_arabert_labels = list(train_df_temp['label'].dropna().unique())
    arabert_labels_list = [dialect_name_mapping.get(str(l), str(l)) for l in raw_arabert_labels]
except Exception as e:
    arabert_labels_list = ['القاهرة (مصر) 🇪🇬', 'تونس 🇹🇳', 'بيروت (لبنان) 🇱🇧', 'الدوحة / الخليج 🇶🇦', 'اللغة العربية الفصحى AR']

marbert_labels_list = ['بيروت (لبنان) 🇱🇧', 'القاهرة (مصر) 🇪🇬', 'الدوحة / الخليج 🇶🇦', 'اللغة العربية الفصحى AR', 'تونس 🇹🇳']

@st.cache_resource
def load_specific_model(model_type):
    if model_type == "arabert":
        model_path = "/content/drive/MyDrive/Arabert_dialect_final_model"
        base_model = "aubmindlab/bert-base-arabertv02"
    else:
        model_path = "/content/drive/MyDrive/marbert_best_checkpoint"
        base_model = "UBC-NLP/MARBERT"

    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=5)

    return tokenizer, model

st.sidebar.header("⚙️ إعدادات المنصة")
model_choice = st.sidebar.selectbox(
    "اختر نموذج التحليل:",
    ("MarBERTv2", "AraBERT")
)

is_arabert = "AraBERT" in model_choice
active_tokenizer, active_model = load_specific_model("arabert" if is_arabert else "marbert")
current_labels = arabert_labels_list if is_arabert else marbert_labels_list

st.sidebar.markdown(
    """
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); padding: 14px; border-radius: 12px; color: #ffffff; font-size: 15px; font-weight: bold; margin-top: 15px; text-align: center; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.4);">
        ✨ النموذج جاهز للتحليل الفوري بكفاءة عالية 🚀
    </div>
    """, 
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🎯 تحليل نص فردي", "⚖️ مقارنة مباشرة بين النموذجين", "📁 تحليل ملف (Batch)"])

with tab1:
    user_input = st.text_area("✍️ أدخل النص هنا لتحليل اللهجة:", placeholder="مثال: عامل إيه يا صاحبي في الشغل؟ أو برشا خدمة...", height=130)

    if st.button("🚀 ابدأ تحليل وتوقع اللهجة الآن"):
        if user_input.strip() == "":
            st.warning("⚠️ يرجى كتابة نص أولاً لنبدأ السحر!")
        else:
            with st.spinner('⏳ جاري المعالجة وتحليل اللهجة بدقة فائقة...'):
                inputs = active_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
                active_model.eval()
                with torch.no_grad():
                    outputs = active_model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)[0]
                    predicted_class_id = outputs.logits.argmax(axis=1).item()

                best_dialect = current_labels[predicted_class_id] if predicted_class_id < len(current_labels) else "غير معروف"
                confidence = probabilities[predicted_class_id].item() * 100

            st.markdown(
                f"""
                <div class="result-box">
                    <h3 style="margin-top: 0; color: #4f46e5;">📊 نتائج التحليل الذكي:</h3>
                    <p style="font-size: 19px;">🎯 <b>اللهجة المتوقعة:</b> <span class="highlight-dialect">{best_dialect}</span></p>
                    <p style="font-size: 18px;">📈 <b>نسبة الثقة:</b> <span class="highlight-conf">{confidence:.2f}%</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

with tab2:
    st.markdown("### ⚖️ مقارنة أداء النموذجين (AraBERT vs MarBERT) في نفس اللحظة")
    comp_input = st.text_area("✍️ أدخل جملة لاختبارها على كلا النموذجين:", placeholder="اكتب جملة هنا...", height=100, key="comp_box")
    
    if st.button("🔍 قارن النتائج الآن"):
        if comp_input.strip() == "":
            st.warning("⚠️ يرجى كتابة نص للمقارنة!")
        else:
            col_a, col_b = st.columns(2)
            
            tok_a, mod_a = load_specific_model("arabert")
            inputs_a = tok_a(comp_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                out_a = mod_a(**inputs_a)
                prob_a = torch.softmax(out_a.logits, dim=1)[0]
                pred_a = out_a.logits.argmax(axis=1).item()
            res_a = arabert_labels_list[pred_a] if pred_a < len(arabert_labels_list) else "غير معروف"
            conf_a = prob_a[pred_a].item() * 100

            tok_m, mod_m = load_specific_model("marbert")
            inputs_m = tok_m(comp_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                out_m = mod_m(**inputs_m)
                prob_m = torch.softmax(out_m.logits, dim=1)[0]
                pred_m = out_m.logits.argmax(axis=1).item()
            res_m = marbert_labels_list[pred_m] if pred_m < len(marbert_labels_list) else "غير معروف"
            conf_m = prob_m[pred_m].item() * 100

            with col_a:
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 20px; border-radius: 15px; border: 2px solid #3b82f6;">
                        <h4 style="color: #60a5fa; margin-top: 0;">🤖 نموذج AraBERT</h4>
                        <p>🎯 اللهجة: <b>{res_a}</b></p>
                        <p>📈 الثقة: <b>{conf_a:.2f}%</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            with col_b:
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 20px; border-radius: 15px; border: 2px solid #ec4899;">
                        <h4 style="color: #f472b6; margin-top: 0;">🤖 نموذج MarBERT</h4>
                        <p>🎯 اللهجة: <b>{res_m}</b></p>
                        <p>📈 الثقة: <b>{conf_m:.2f}%</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

with tab3:
    st.markdown("### 📁 تحليل مجموعة جمل دفعة واحدة (Batch Analysis)")
    st.markdown("قومي برفع ملف CSV يحتوي على عمود يحمل اسم `text` يضم عدة جمل لتصنيفها دفعة واحدة.")
    
    uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            st.success(f"✅ تم رفع الملف بنجاح ويحتوي على {len(df)} جملة.")
            if st.button("⚡ ابدأ تحليل الملف"):
                tok_m, mod_m = load_specific_model("marbert")
                predictions = []
                confidences = []
                
                with st.spinner('⏳ جاري تحليل الجمل دفعة واحدة...'):
                    for text in df['text']:
                        inputs = tok_m(str(text), return_tensors="pt", padding=True, truncation=True, max_length=128)
                        with torch.no_grad():
                            outputs = mod_m(**inputs)
                            probs = torch.softmax(outputs.logits, dim=1)[0]
                            pred_id = outputs.logits.argmax(axis=1).item()
                        predictions.append(marbert_labels_list[pred_id] if pred_id < len(marbert_labels_list) else "غير معروف")
                        confidences.append(f"{probs[pred_id].item() * 100:.2f}%")
                
                df['اللهجة المتوقعة'] = predictions
                df['نسبة الثقة'] = confidences
                st.dataframe(df)
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 تحميل النتائج كملف CSV",
                    data=csv_data,
                    file_name='dialect_analysis_results.csv',
                    mime='text/csv',
                )
        else:
            st.error("⚠️ ملف الـ CSV يجب أن يحتوي على عمود باسم `text` يحتوي على الجمل المراد تحليلها!")

st.markdown("<br><hr style='border-color: #334155;'><p class='footer-text'>تم تطوير المنصة بعناية فائقة لدعم التحليل اللغوي المتقدم 🚀</p>", unsafe_allow_html=True)
