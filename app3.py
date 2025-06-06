import streamlit as st
import pandas as pd
import io
import time
import multiprocessing as mp
import base64
from intellimatch_controller import IntelliMatchController

st.set_page_config(page_title="IntelliMatch Demo", layout="wide")
st.title("🧠 IntelliMatch: Company Matching Demo")
st.markdown("IntelliMatch is an end-to-end Streamlit demo that ingests your company lists, runs K-Fold TF-IDF, "
            "NLP Preprocessing, Keyword Classification, and ML Fuzzy Matching to accurately enrich and match records between"
            "the two sources with interactive per-stage previews, demo workflows and downloadable Excel output.")
st.markdown("### Get started: \n"
            "1. Make sure your input Excel file has the following columns: Company, Source[source1, source2]\n"
            "2. You can set Preview sample size and Parallel process count parameters as required.\n"
            "2. After uploading input file, click on Start Pipeline.\n"
            "3. At each stage, a sample workflow example will be visible as you wait for the stage to complete.\n"
            "4. Download results in Excel file once the pipeline is executed.")

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
    file_ = open("/Users/sauravdosi/Documents/intellimatch/img/tfidf.gif", "rb")
    contents = file_.read()
    data_url_tfidf = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_tfidf}" alt="TF-IDF Workflow">',
        unsafe_allow_html=True,
    )

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
    ex = example_row(st.session_state.df_tfidf)
    if ex is not None and "tfidf" in ex.columns:
        st.subheader("Example TF-IDF Weights")
        # assume normalized_tfidf is dict
        weights = ex["normalized_tfidf"].values[0]
        st.bar_chart(pd.Series(weights).sort_values(ascending=False).head(20))
    if st.button("▶ Next: NLP Preprocessing"):
        st.session_state.controller.df = pd.concat([
            st.session_state.df_tfidf[
                (st.session_state.df_tfidf["Source"] == st.session_state.controller.reference_column)
                & (st.session_state.df_tfidf["HasOwnerRole"] == 1)
            ],
            st.session_state.df_tfidf[st.session_state.df_tfidf["Source"] == "Salesforce"]
        ])
        st.session_state.stage = 2

# Stage 2: NLP Preprocessing
if st.session_state.stage >= 2:
    st.header("2️⃣ NLP Preprocessing")

    file_ = open("/Users/sauravdosi/Documents/intellimatch/img/nlp_preprocess.gif", "rb")
    contents = file_.read()
    data_url_nlp = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_nlp}" alt="NLP Preprocessing Workflow">',
        unsafe_allow_html=True,
    )

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

# Stage 3: Keyword Classification
if st.session_state.stage >= 3:
    st.header("3️⃣ Keyword Classification")

    file_ = open("/Users/sauravdosi/Documents/intellimatch/img/keyword_class.gif", "rb")
    contents = file_.read()
    data_url_kw = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_kw}" alt="Keyword Classifier Workflow">',
        unsafe_allow_html=True,
    )

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
        st.write("Predicted Labels:", ex["Predicted Labels"].values[0])
        st.write("Predicted Label Names:", ex["Predicted Label Names"].values[0])
    if st.button("▶ Next: ML Fuzzy Matching"):
        st.session_state.controller.df = st.session_state.df_kw
        st.session_state.stage = 4

# Stage 4: ML Fuzzy Matching
if st.session_state.stage >= 4:
    st.header("4️⃣ ML Fuzzy Matching")

    file_ = open("/Users/sauravdosi/Documents/intellimatch/img/ml_fuzzy.gif", "rb")
    contents = file_.read()
    data_url_fuzzy = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_fuzzy}" alt="ML Fuzzy Matching Workflow">',
        unsafe_allow_html=True,
    )

    if st.session_state.df_fuzzy is None:
        start = time.time()
        with st.spinner("Running fuzzy matching…"):
            st.session_state.df_fuzzy = st.session_state.controller.ml_fuzzy_matching_run()
        st.session_state.timings["fuzzy"] = time.time() - start
        st.success(f"Done in {st.session_state.timings['fuzzy']:.1f}s")

    st.dataframe(st.session_state.df_fuzzy.head(sample_size))

    ex = example_row(st.session_state.df_fuzzy)
    if ex is not None and "Match Score" in ex:
        st.subheader("Example Fuzzy Match")
        st.write("Matched Company:", ex["Matched Company Name"].values[0])
        st.write("Score:", ex["Match Score"].values[0])
        # if there's a dict column, show it
        if "Matched Company Labels Dict" in ex:
            st.write(ex["Matched Company Labels Dict"].values[0])
    if st.button("▶ Next: Postprocessing"):
        st.session_state.controller.df = st.session_state.df_fuzzy
        st.session_state.stage = 5

# Stage 5: Postprocessing
if st.session_state.stage >= 5:
    st.header("5️⃣ Postprocessing")

    file_ = open("/Users/sauravdosi/Documents/intellimatch/img/postprocess.gif", "rb")
    contents = file_.read()
    data_url_post = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url_post}" alt="Postprocess Workflow">',
        unsafe_allow_html=True,
    )

    if st.session_state.df_post is None:
        start = time.time()
        with st.spinner("Postprocessing…"):
            st.session_state.df_post = st.session_state.controller.postprocess_run()
        st.session_state.timings["post"] = time.time() - start
        st.success(f"Done in {st.session_state.timings['post']:.1f}s")

    st.dataframe(st.session_state.df_post.head(sample_size))

    if st.button("▶ Next: Final Results"):
        st.session_state.controller.df = st.session_state.df_post
        st.session_state.stage = 6

# Stage 6: Final Results & Download
if st.session_state.stage >= 6:
    st.header("6️⃣ Final Results")
    if st.session_state.final_df is None:
        st.session_state.final_df = st.session_state.controller.df
    st.dataframe(st.session_state.final_df.head(sample_size))
    # show total pipeline time
    total = sum(st.session_state.timings.values())
    st.write(f"**Total time:** {total:.1f}s")
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
