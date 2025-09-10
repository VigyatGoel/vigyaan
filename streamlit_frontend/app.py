import streamlit as st
import pandas as pd

import config
from api_client import MLPlatformAPI
from session_state import initialize_session_state
from views.overview import display_data_overview
from views.visualizations import display_visualizations
from views.analysis import display_data_analysis
from views.machine_learning import display_machine_learning

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)


def main():
    """Main Streamlit application"""
    st.markdown(
        """
    <style>
    /* Better spacing and styling */
    .nav-header {
        margin-bottom: 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    initialize_session_state()

    api_client = MLPlatformAPI(config.API_BASE_URL)

    # Native Streamlit title at top
    st.title("🤖 Machine Learning & Data Science Platform")
    st.markdown("---")

    with st.sidebar:
        st.header("📁 File Upload")

        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], help="Upload your dataset in CSV format"
        )

        if uploaded_file is not None and st.session_state.uploaded_file_path is None:
            with st.spinner("Uploading file..."):
                upload_result = api_client.upload_csv(uploaded_file)

                if upload_result:
                    st.session_state.uploaded_file_path = upload_result[
                        "absolute_file_path"
                    ]
                    st.session_state.original_filename = upload_result.get(
                        "original_filename", uploaded_file.name
                    )
                    st.success(
                        f"✅ File '{st.session_state.original_filename}' uploaded successfully to temporary location"
                    )

                    with st.spinner("Extracting features..."):
                        features_result = api_client.extract_features(
                            st.session_state.uploaded_file_path
                        )

                        if features_result:
                            st.session_state.features = features_result[
                                "feature_columns"
                            ]
                            st.session_state.label_column = features_result[
                                "label_column"
                            ]
                            st.session_state.data_loaded = True

                            try:
                                uploaded_file.seek(0)
                                st.session_state.df = pd.read_csv(uploaded_file)
                            except Exception as e:
                                st.error(
                                    f"Error reading CSV from uploaded file: {str(e)}"
                                )

    if not st.session_state.data_loaded:
        st.info("Please upload a CSV file to get started")

        st.subheader("🎯 What you can do with this platform:")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            **📊 Data Overview**
            - File information
            - Data description
            - Statistical summary
            """)

        with col2:
            st.markdown("""
            **📈 Visualizations**
            - Scatter plots
            - Histograms
            - Line plots
            - Correlation matrix
            - Box plots
            - Pair plots
            - Area plots
            """)

        with col3:
            st.markdown("""
            **🔍 Data Analysis**
            - Feature exploration
            - Data quality checks
            - Statistical insights
            """)

        with col4:
            st.markdown("""
            **🤖 Machine Learning**
            - Train ML models
            - Compare performance
            - Download trained models
            - Get template code
            """)

    else:
        tab_overview, tab_viz, tab_analysis, tab_ml = st.tabs(
            [
                "📊 Data Overview",
                "📈 Visualizations",
                "🔍 Data Analysis",
                "🤖 Machine Learning",
            ]
        )

        with tab_overview:
            display_data_overview(api_client)
        with tab_viz:
            display_visualizations(api_client)
        with tab_analysis:
            display_data_analysis(api_client)
        with tab_ml:
            display_machine_learning(api_client)


if __name__ == "__main__":
    main()
