# app.py
import streamlit as st
import pandas as pd
import io
import multiprocessing as mp
from intellimatch_controller import IntelliMatchController

st.set_page_config(page_title="IntelliMatch Demo", layout="wide")
st.title("🧠 IntelliMatch: Company Matching Demo")
st.markdown(
    "Upload your company list, configure parameters, and click **Start** to run the pipeline stage-by-stage."
)

# — Sidebar inputs
st.sidebar.header("Configuration")
sample_size = st.sidebar.number_input("Preview sample size", 10, 1000, 100, 10)
n_process = st.sidebar.slider("Parallel processes", 1, mp.cpu_count(), 1)

st.sidebar.header("Input File")
input_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

# — Initialize session state
if "stage" not in st.session_state:
    st.session_state.stage = 0
    st.session_state.controller = None
    st.session_state.df_tfidf = None
    st.session_state.df_nlp = None
    st.session_state.df_kw = None
    st.session_state.df_fuzzy = None
    st.session_state.df_post = None
    st.session_state.final_df = None

# — Start button
if st.sidebar.button("▶ Start"):
    if not input_file:
        st.sidebar.error("Please upload your input Excel first.")
    else:
        # Load the DataFrame and instantiate controller
        df = pd.read_excel(input_file)
        st.session_state.controller = IntelliMatchController(
            input_path=df,
            output_path="data/intellimatch_results.xlsx",
            n_process=n_process
        )
        st.session_state.controller.df = df
        st.session_state.stage = 1

# — Stage 1: K-Fold TF-IDF
if st.session_state.stage >= 1 and st.session_state.controller:
    st.header("1️⃣ K-Fold TF-IDF Generation")
    if st.session_state.df_tfidf is None:
        with st.spinner("Running K-Fold TF-IDF…"):
            kfg = st.session_state.controller.kfold_tfidf_generator_driver
            # kfg.df = st.session_state.controller.df.copy()
            st.session_state.df_tfidf = kfg.run()
        st.success("Done.")
    # show preview
    st.dataframe(st.session_state.df_tfidf[:sample_size])

    ref = st.session_state.controller.reference_column
    df1 = st.session_state.df_tfidf[
        (st.session_state.df_tfidf["Source"] == ref)
        & (st.session_state.df_tfidf["HasOwnerRole"] == 1)
    ].head(sample_size)
    df2 = st.session_state.df_tfidf[
        st.session_state.df_tfidf["Source"] == "Salesforce"
    ].head(sample_size)

    if st.button("▶ Next: NLP Preprocessing"):
        st.session_state.controller.df = pd.concat([df1, df2])
        st.session_state.stage = 2

# — Stage 2: NLP Preprocessing
if st.session_state.stage >= 2:
    st.header("2️⃣ NLP Preprocessing")
    if st.session_state.df_nlp is None:
        with st.spinner("Running NLP Preprocessing…"):
            st.session_state.df_nlp = st.session_state.controller.nlp_preprocessing_run()
        st.success("Done.")
    # lengths = st.session_state.df_nlp["cleaned_words"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)
    # st.bar_chart(lengths.value_counts().sort_index())
    st.dataframe(st.session_state.df_nlp[:sample_size])

    if st.button("▶ Next: Keyword Classification"):
        st.session_state.controller.df = st.session_state.df_nlp
        st.session_state.stage = 3

# — Stage 3: Keyword Classification
if st.session_state.stage >= 3:
    st.header("3️⃣ Keyword Classification")
    if st.session_state.df_kw is None:
        with st.spinner("Running Keyword Classifier…"):
            st.session_state.df_kw = st.session_state.controller.keyword_classifier_run()
        st.success("Done.")
    # st.bar_chart(st.session_state.df_kw["Predicted Label Names"].value_counts())
    st.dataframe(st.session_state.df_kw[:sample_size])
    if st.button("▶ Next: ML Fuzzy Matching"):
        st.session_state.controller.df = st.session_state.df_kw
        st.session_state.stage = 4

# — Stage 4: ML Fuzzy Matching
if st.session_state.stage >= 4:
    st.header("4️⃣ ML Fuzzy Matching")
    if st.session_state.df_fuzzy is None:
        with st.spinner("Running ML Fuzzy Matching…"):
            st.session_state.df_fuzzy = st.session_state.controller.ml_fuzzy_matching_run()
        st.success("Done.")
    # st.bar_chart(st.session_state.df_fuzzy["Match Score"].value_counts(bins=20))
    st.dataframe(st.session_state.df_fuzzy[:sample_size])
    if st.button("▶ Next: Postprocessing"):
        st.session_state.controller.df = st.session_state.df_fuzzy
        st.session_state.stage = 5

# — Stage 5: Postprocessing
if st.session_state.stage >= 5:
    st.header("5️⃣ Postprocessing")
    if st.session_state.df_post is None:
        with st.spinner("Running Postprocessing…"):
            st.session_state.df_post = st.session_state.controller.postprocess_run()
        st.success("Done.")
    st.dataframe(st.session_state.df_post.head(sample_size))
    if st.button("▶ Next: Final Results"):
        st.session_state.controller.df = st.session_state.df_post
        st.session_state.stage = 6

# — Stage 6: Final Results & Download
if st.session_state.stage >= 6:
    st.header("6️⃣ Final Results")
    if st.session_state.final_df is None:
        with st.spinner("Finalizing…"):
            # use execute_pipeline here to get the last step if needed;
            # but since df_post is your last stage, we can just assign:
            st.session_state.final_df = st.session_state.controller.df
        st.success("Pipeline complete!")
    st.dataframe(st.session_state.final_df.head(sample_size))

    # Download button
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        st.session_state.final_df.to_excel(writer, index=False, sheet_name="Results")
    buf.seek(0)
    st.download_button(
        "📥 Download Full Excel",
        data=buf,
        file_name="intellimatch_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
