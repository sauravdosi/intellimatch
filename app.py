import streamlit as st
import pandas as pd
import io
import time
import multiprocessing as mp
import base64
from streamlit_lottie import st_lottie
import requests
from PIL import Image
from intellimatch_controller import IntelliMatchController

st.set_page_config(page_title="🧠 IntelliMatch", layout="wide")

# define a mapping from label → colour (hex or CSS name)
LABEL_COLORS = {
    "LabelA": "#8dd3c7",
    "LabelB": "#ffffb3",
    "LabelC": "#bebada",
    # …add all of your labels here…
}

# ——————————————————————————————————————————
# 🎆 Animated Header
# ——————————————————————————————————————————
@st.cache_data
def load_lottie_url(url):
    r = requests.get(url)
    return r.json()

lottie_brain = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

col1, col2 = st.columns([1, 3], gap="small")

with col1:
    st_lottie(lottie_brain, height=150, key="brain")

with col2:
    st.markdown(
        """
        <style>
          @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to   { opacity: 1; transform: translateY(0); }
          }
          .intellimatch-title h1,
          .intellimatch-title p {
            color: var(--text-color);
            transition: transform 0.3s ease;
          }
          .intellimatch-title h1:hover {
            transform: scale(1.05);
          }
          .intellimatch-title p:hover {
            transform: scale(1.02);
          }
        </style>

        <div class="intellimatch-title" style="animation: fadeInDown 1s ease-out; text-align:left;">
          <h1 style="font-size:3em; margin:0;">🧠 IntelliMatch</h1>
          <p style="font-size:1.3em; margin-top:0.3em;">
            An end-to-end company name matching demo with live animations 🎉
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
# ——————————————————————————————————————————
# 🌟 One-liner
# ——————————————————————————————————————————
st.info("**IntelliMatch** is an end-to-end Streamlit demo that ingests your company lists, runs **K-Fold TF-IDF**, "
            "**NLP Preprocessing**, **Keyword Classification**, and **ML Fuzzy Matching** to accurately enrich and match records between"
            "the two sources with interactive per-stage previews, demo workflows and downloadable Excel report! 🚀")
# ——————————————————————————————————————————
# 📚 Get Started (Emoji-rich)
# ——————————————————————————————————————————
with st.expander("🛠️ How to Get Started"):
    st.markdown("""
    1. 🗂️ **Prepare your Excel**  
       • **Company** column (e.g. “Acme Corp”)  
       • **Source** column (`inference`, `reference`) - finds one reference company name for every inference company name

    2. ⚙️ **Configure** in the sidebar  
       • **🔎 Preview size:** how many rows to peek at  
       • **⚡ Parallel processes:** tune for your machine

    3. 📤 **Upload & Start**  
       • Click **Upload** and then **▶ Start Pipeline**

    4. 👀 **Watch it happen**  
       • At each of the 6 stages you’ll see:  
         – A sample table or chart  
         – An animated GIF of the mini-workflow  
         – Timing info ⏱️

    5. 📥 **Download** your final Excel report  
       • Fully enriched, matched & ready to share!
    """)


# Sidebar
st.sidebar.header("⚙️ Configuration")
sample_size = st.sidebar.number_input("Preview sample size", 10, 1000, 100, 10)
n_process = st.sidebar.slider("Parallel processes", 1, mp.cpu_count(), 1)
input_file = st.sidebar.file_uploader("Upload Input Excel", type=["xlsx"])

# Session state init
if "stage" not in st.session_state:
    for key in ["df_tfidf","df_nlp","df_kw","df_fuzzy","df_post","final_df","timings","controller"]:
        st.session_state[key] = None
    st.session_state.stage = 0
    st.session_state.timings = {}

# Start
if st.sidebar.button("▶ Start Pipeline"):
    if not input_file:
        st.sidebar.error("Please upload your input file.")
    else:
        df = pd.read_excel(input_file)
        ctrl = IntelliMatchController(input_path=df,
                                      reference_path="data/tfidf_reference.json",
                                      output_path="data/intellimatch_results.xlsx",
                                      n_process=n_process)
        ctrl.df = df
        st.session_state.controller = ctrl
        st.session_state.stage = 1

# Helper to pick example row
def example_row(df):
    return df.iloc[0:1].copy() if df is not None and len(df)>0 else None

# Stage 1: TF-IDF
if st.session_state.stage >= 1:
    st.header("1️⃣ K-Fold TF-IDF Generation")
    file_ = open("img/tfidf.gif", "rb")
    contents = file_.read()
    data_url_tfidf = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_tfidf}" alt="TF-IDF Workflow">',
        unsafe_allow_html=True,
    )
    # you can also add an extra blank line if needed:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.df_tfidf is None:
        start = time.time()
        with st.spinner("Computing TF-IDF…"):
            kfg = st.session_state.controller.kfold_tfidf_generator_driver
            # kfg.df = st.session_state.controller.df.copy()
            st.session_state.df_tfidf = kfg.run()
        st.session_state.timings["tfidf"] = time.time() - start
        st.success(f"Done in {st.session_state.timings['tfidf']:.1f}s")
    # show sample table
    st.dataframe(st.session_state.df_tfidf.head(sample_size))

    # example visualization
    # ex = example_row(st.session_state.df_tfidf)
    # if ex is not None and "tfidf" in ex.columns:
    #     st.subheader("Example TF-IDF Weights")
    #     # assume normalized_tfidf is dict
    #     weights = ex["normalized_tfidf"].values[0]
    #     st.bar_chart(pd.Series(weights).sort_values(ascending=False).head(20))
    if st.button("▶ Next: NLP Preprocessing"):
        st.session_state.controller.df = pd.concat([
            st.session_state.df_tfidf[
                (st.session_state.df_tfidf["Source"] == st.session_state.controller.reference_column)
            ],
            st.session_state.df_tfidf[st.session_state.df_tfidf["Source"] == st.session_state.controller.inference_column]
        ])
        st.session_state.stage = 2

# Stage 2: NLP Preprocessing
if st.session_state.stage >= 2:
    st.header("2️⃣ NLP Preprocessing")

    file_ = open("img/nlp_preprocess.gif", "rb")
    contents = file_.read()
    data_url_nlp = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_nlp}" alt="NLP Preprocessing Workflow">',
        unsafe_allow_html=True,
    )
    # you can also add an extra blank line if needed:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.df_nlp is None:
        start = time.time()
        with st.spinner("Tokenizing & cleaning…"):
            st.session_state.df_nlp = st.session_state.controller.nlp_preprocessing_run()
        st.session_state.timings["nlp"] = time.time() - start
        st.success(f"Done in {st.session_state.timings['nlp']:.1f}s")

    st.dataframe(st.session_state.df_nlp.head(sample_size))

    ex = example_row(st.session_state.df_nlp)
    if ex is not None and "cleaned_words" in ex.columns:
        words = ex["cleaned_words"].values[0]
        st.subheader("Example Tokens")
        st.write(words)
        st.write(f"Token count: {len(words)}")
    if st.button("▶ Next: Keyword Classification"):
        st.session_state.controller.df = st.session_state.df_nlp
        st.session_state.stage = 3

# 1) Define your label→colour map
LABEL_COLORS = {
    "Important": "#ffc0cb",
    "Subsidiary": "#add8e6",
    "Generic": "#ffffb3",
    # …add all your labels here…
}

def render_keyword_tiles(row, text_col="words", label_col="Predicted Label Names"):
    # row[text_col]  : list of keyword strings
    # row[label_col]: list of corresponding label strings
    html = "<div style='display:flex; flex-wrap: wrap; gap:8px; margin-bottom:12px;'>"
    for kw, lbl in zip(row[text_col], row[label_col]):
        col = LABEL_COLORS.get(lbl, "#dddddd")
        html += (
            f"<span style='"
            f"display:inline-block; "
            f"padding:6px 12px; "
            f"background:{col}; "
            f"border-radius:12px; "
            f"font-size:0.95em; "
            f"line-height:1; "
            f"'>"
            f"{kw}"
            f"</span>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_legend():
    # builds a little legend showing each label swatch
    legend_html = "<div style='display:flex; gap:16px; flex-wrap: wrap; margin-bottom:20px;'>"
    for lbl, col in LABEL_COLORS.items():
        legend_html += (
            "<div style='display:flex; align-items:center; gap:4px;'>"
            f"<span style='width:14px; height:14px; background:{col}; display:inline-block; border-radius:3px;'></span>"
            f"<span style='font-size:0.9em;'>{lbl}</span>"
            "</div>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


# Stage 3: Keyword Classification
if st.session_state.stage >= 3:
    st.header("3️⃣ Keyword Classification")

    file_ = open("img/keyword_class.gif", "rb")
    contents = file_.read()
    data_url_kw = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_kw}" alt="Keyword Classifier Workflow">',
        unsafe_allow_html=True,
    )
    # you can also add an extra blank line if needed:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.df_kw is None:
        start = time.time()
        with st.spinner("Classifying keywords…"):
            st.session_state.df_kw = st.session_state.controller.keyword_classifier_run()
        st.session_state.timings["kw"] = time.time() - start
        st.success(f"Done in {st.session_state.timings['kw']:.1f}s")

    st.dataframe(st.session_state.df_kw.head(sample_size))

    ex = example_row(st.session_state.df_kw)
    if ex is not None:
        st.subheader("Example Prediction")
        # 2) Render the tiles
        render_keyword_tiles(ex.iloc[0])
        # 3) Render the legend under the tiles
        render_legend()

        # 4) Add extra gap before the button
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    if st.button("▶ Next: ML Fuzzy Matching"):
        st.session_state.controller.df = st.session_state.df_kw
        st.session_state.stage = 4

# Stage 4: ML Fuzzy Matching
if st.session_state.stage >= 4:
    st.header("4️⃣ ML Fuzzy Matching")

    file_ = open("img/ml_fuzzy.gif", "rb")
    contents = file_.read()
    data_url_fuzzy = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_fuzzy}" alt="ML Fuzzy Matching Workflow">',
        unsafe_allow_html=True,
    )
    # you can also add an extra blank line if needed:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.df_fuzzy is None:
        start = time.time()
        with st.spinner("Running fuzzy matching…"):
            st.session_state.df_fuzzy = st.session_state.controller.ml_fuzzy_matching_run()
        st.session_state.timings["fuzzy"] = time.time() - start
        st.success(f"Done in {st.session_state.timings['fuzzy']:.1f}s")

    st.dataframe(st.session_state.df_fuzzy.head(sample_size))

    # ---- new top‐example logic ----
    df = st.session_state.df_fuzzy
    if df is not None and "Match Score" in df.columns:
        # 1) grab the one row with the highest match score
        top = df.sort_values("Match Score", ascending=False).head(1)
        if not top.empty:
            ex = top.iloc[0]
            st.subheader("Top Fuzzy Match Example")
            st.write("**Inference Company Name:**", ex.get("AHC Company Name", "—"))
            st.write("**Matched Company Name:**", ex.get("Matched Company Name", "—"))
            st.write("**Match Score:**", ex["Match Score"])
            st.write("**Match Category:**", ex["Match Category"])
            # if you have extra data to show (e.g. labels dict), you can still include it:
            # if "Matched Company Labels Dict" in ex:
            #     st.write("Labels Dict:", ex["Matched Company Labels Dict"])

    # ---- end top‐example logic ----
    if st.button("▶ Next: Postprocessing"):
        st.session_state.controller.df = st.session_state.df_fuzzy
        st.session_state.stage = 5

# Stage 5: Postprocessing
if st.session_state.stage >= 5:
    st.header("5️⃣ Postprocessing")

    file_ = open("img/postprocess.gif", "rb")
    contents = file_.read()
    data_url_post = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_post}" alt="Postprocess Workflow">',
        unsafe_allow_html=True,
    )
    # you can also add an extra blank line if needed:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.df_post is None:
        start = time.time()
        with st.spinner("Postprocessing…"):
            st.session_state.df_post = st.session_state.controller.postprocess_run()
        st.session_state.timings["post"] = time.time() - start
        st.success(f"Done in {st.session_state.timings['post']:.1f}s")

    st.dataframe(st.session_state.df_post.head(sample_size))
    # show total pipeline time
    total = sum(st.session_state.timings.values())
    st.write(f"**Total time:** {total:.1f}s")

    if st.session_state.final_df is None:
        st.session_state.final_df = st.session_state.controller.df

    # Download
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        st.session_state.final_df.to_excel(w, index=False, sheet_name="Results")
    buf.seek(0)
    st.download_button(
        "📥 Download Full Excel",
        data=buf,
        file_name="intellimatch_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
    )

    # if st.button("▶ Next: Final Results"):
    #     st.session_state.controller.df = st.session_state.df_post
    #     st.session_state.stage = 6

# # Stage 6: Final Results & Download
# if st.session_state.stage >= 6:
#     st.header("6️⃣ Final Results")
#     if st.session_state.final_df is None:
#         st.session_state.final_df = st.session_state.controller.df
#     st.dataframe(st.session_state.final_df.head(sample_size))
#     # show total pipeline time
#     total = sum(st.session_state.timings.values())
#     st.write(f"**Total time:** {total:.1f}s")
#     # Download
#     buf = io.BytesIO()
#     with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
#         st.session_state.final_df.to_excel(w, index=False, sheet_name="Results")
#     buf.seek(0)
#     st.download_button(
#         "📥 Download Full Excel",
#         data=buf,
#         file_name="intellimatch_results.xlsx",
#         mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
#     )
