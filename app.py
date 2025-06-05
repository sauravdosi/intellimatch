import streamlit as st
import pandas as pd
import io
import multiprocessing as mp
from intellimatch_controller import IntelliMatchController

st.set_page_config(
    page_title="IntelliMatch Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 IntelliMatch: Company Matching Demo")
st.markdown("Upload your company list and active HCs, configure parameters, and click **Generate** to run the full pipeline.")

# Sidebar configuration
st.sidebar.header("Configuration")
sample_size = st.sidebar.number_input("Sample size for preview", 10, 1000, 100, 10)
n_process = st.sidebar.slider("Parallel processes", 1, mp.cpu_count(), 1)

# File uploaders
st.sidebar.header("Input Files")
input_file = st.sidebar.file_uploader("Input Excel (.xlsx)", type=["xlsx"])
# active_hcs_file = st.sidebar.file_uploader("Active HCs Excel (.xlsx)", type=["xlsx"])

# Generate button
generate = st.sidebar.button("🚀 Generate")

# print(input_file)

if generate:
    if not input_file:
            # or not active_hcs_file):
        st.sidebar.error("Please upload the input file.")
    else:
        df = pd.read_excel(input_file)
        hcs_df = pd.read_excel("data/isn_active_hcs.xlsx")

        print("Input DF:")
        print(df)

        # Initialize controller and inject data
        controller = IntelliMatchController(
            input_path=df,
            output_path="data/intellimatch_results.xlsx",
            n_process=n_process
        )
        controller.df = df
        controller.active_hcs_df = hcs_df

        # Execute pipeline
        with st.spinner("Running IntelliMatch pipeline..."):
            final_df = controller.execute_pipeline()
        st.success("Pipeline completed")

        # Preview results
        st.subheader("Final Results Preview")
        st.dataframe(final_df.head(sample_size))

        # Download Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False, sheet_name="Results")
        buffer.seek(0)
        st.download_button(
            "📥 Download Results",
            data=buffer,
            file_name="data/intellimatch_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Upload files and click 🚀 Generate to start the workflow.")
