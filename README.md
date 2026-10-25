# 🌍 Fine-Grained Arabic Dialect Classification Platform
### End-to-End NLP Graduation Project | Classical ML Stacking vs. AraBERTv02 vs. MARBERTv2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 1. Project Overview
Arabic is characterized by **Diglossia**, where Modern Standard Arabic (MSA) coexists alongside diverse regional dialects that lack standardized orthography and share extensive lexical overlap. 

This project implements a complete, production-ready **Natural Language Processing (NLP)** pipeline to classify Arabic text across **5 major dialectal varieties** using the **MADAR Corpus**:
* 🇱🇧 **Beirut** *(Levantine Arabic)*
* 🇪🇬 **Cairo** *(Egyptian Arabic)*
* 🇶🇦 **Doha** *(Gulf Arabic)*
* 📖 **MSA** *(Modern Standard Arabic)*
* 🇹🇳 **Tunis** *(Maghrebi Arabic)*

We systematically benchmark **Classical Machine Learning Ensembles (TF-IDF + Stacking)** against state-of-the-art pretrained Arabic Transformer architectures (**AraBERTv02** and **MARBERTv2**), culminating in a real-time interactive **Streamlit Web Platform**.

---

## 🏆 2. Final Benchmark Results

All models were trained and evaluated on a strictly stratified split (**70% Train / 15% Validation / 15% Unseen Test**) with **zero data leakage**:

| Model Architecture | Feature Representation | Validation Accuracy | Unseen Test Accuracy | Macro F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classical Stacking Ensemble** *(LinearSVC + LogisticRegression + MultinomialNB + SGD)* | Hybrid Word `(1,3)` & Char_wb `(2,5)` TF-IDF (`150k` features) | `90.85%` | `90.93%` | `90.92%` | Strong Baseline |
| **AraBERTv02** *(Fine-Tuned)* | Subword WordPiece Embeddings (`aubmindlab/bert-base-arabertv02`) | `91.03%` | — | `91.00%` | Deep Learning Contender |
| **MARBERTv2** *(Fine-Tuned)* | Dialect-Aware Contextual Embeddings (`UBC-NLP/MARBERTv2`) | **`92.02%`** | **`92.25%`** | **`92.24%`** | 🏆 **Champion Model** |

### 📊 Per-Dialect Performance (`MARBERTv2` on Unseen Test Set)
* **Tunis 🇹🇳:** Precision `95.21%` | Recall `93.69%` | **F1-Score: `94.44%`**
* **MSA 📖:** Precision `93.60%` | Recall `94.61%` | **F1-Score: `94.10%`**
* **Cairo 🇪🇬:** Precision `92.42%` | Recall `91.47%` | **F1-Score: `91.94%`**
* **Doha 🇶🇦:** Precision `90.02%` | Recall `91.47%` | **F1-Score: `90.74%`**
* **Beirut 🇱🇧:** Precision `90.09%` | Recall `89.99%` | **F1-Score: `90.04%`**

---

## 🧠 3. Key Linguistic & Engineering Insights

1. **Why Stopwords & Short Sentences Were Retained:**
   * Unlike standard topic classification, critical Arabic dialectal markers reside directly inside functional stopwords and interrogatives (e.g., `إيه` vs. `شو` vs. `شنوة`, `مش` vs. `مو`, `عايز` vs. `بدي`). Removing standard Arabic stopwords destroys vital dialectal signals.
2. **Custom Arabic Orthographic Normalization:**
   * Stripped diacritics (`Tashkeel`), elongation (`Tatweel`), URLs, punctuation, and Latin characters.
   * Unified orthographic variants (`أ, إ, آ ➔ ا`, `ى ➔ ي`, `ة ➔ ه`) to minimize vocabulary sparsity across informal user inputs.
3. **Why `MARBERTv2` Outperformed `AraBERTv02` and `Stacking`:**
   * While `AraBERTv02` was pre-trained primarily on formal MSA news corpora, **`MARBERTv2`** was pre-trained on **1 Billion dialectal Arabic tweets**, enabling superior subword tokenization and contextual disambiguation of agglutinated prefixes (e.g., `شورايك`, `هالفيش`, `فماش`).
4. **Error Analysis & Theoretical Dataset Ceiling:**
   * Deep analysis of the `453` misclassified test samples out of `5,845` (`7.75%` error rate) revealed an average sentence length of **3.86 words** for errors compared to **7.21 words** for correct predictions—proving that remaining errors stem from ultra-short, context-deprived phrases shared across neighboring dialects (e.g., `Beirut` vs. `Doha`).

---

## 📂 4. Repository Structure

```text
arabic-dialect-classification/
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb                # Data loading, orthographic cleaning & deduplication
│   ├── 02_EDA_and_Data_Splitting.ipynb          # Exploratory Data Analysis & Stratified 70/15/15 split
│   ├── 03_Baseline_Models.ipynb                 # Hybrid TF-IDF & Classical Stacking Ensemble (90.93%)
│   ├── 04_MARBERT_FineTuning_and_Analysis.ipynb # MARBERTv2 Fine-Tuning (92.25%) & Deep Error Analysis
│   ├── 05_Arabert_dialect_final_model.ipynb     # AraBERTv02 Fine-Tuning & Evaluation (91.03%)
│   └── 06_Streamlit_Interactive_App.ipynb       # Cloud deployment notebook via Streamlit & Ngrok
│
├── streamlit_app.py                             # Production English UI Streamlit Application
├── app.py                                       # Arabic RTL UI Streamlit Application
├── requirements.txt                             # Python dependencies for local/cloud deployment
└── README.md                                    # Project documentation
```

---

## 🚀 5. Interactive Streamlit Web Platform

Our deployed web application (`streamlit_app.py` / `app.py`) provides three real-time inference modes:
1. **🎯 Single Sentence Analysis:** Real-time text normalization, dialect prediction, confidence percentage, smart linguistic neutrality alerts (`< 45%` confidence), and probability progress bars across all 5 dialects.
2. **⚖️ Head-to-Head Model Comparison:** Simultaneous side-by-side evaluation comparing **MARBERTv2** against **AraBERTv02** on tricky dialectal expressions.
3. **📁 Batch CSV Inference:** Upload any CSV file containing a `text` column to classify hundreds of sentences automatically and export UTF-8 results.

### 💻 How to Run Locally
```bash
# 1. Clone the repository
git clone [https://github.com/meedooatef22/arabic-dialect-classification.git](https://github.com/meedooatef22/arabic-dialect-classification.git)
cd arabic-dialect-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit web app
streamlit run streamlit_app.py
```
