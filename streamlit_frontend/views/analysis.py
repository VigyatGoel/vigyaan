import pandas as pd
import plotly.express as px
import streamlit as st


def display_data_analysis(api_client):
    """Display data analysis page"""
    st.header("🔍 Data Analysis")

    if st.session_state.df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔢 Numerical Columns")
            numerical_cols = st.session_state.df.select_dtypes(
                include=["number"]
            ).columns.tolist()

            if numerical_cols:
                for i, col in enumerate(numerical_cols, 1):
                    st.markdown(f"**{i}.** `{col}`")

                st.info(f"Total: {len(numerical_cols)} numerical columns")
            else:
                st.warning("No numerical columns found")

            st.subheader("📊 Missing Values")
            missing_data = st.session_state.df.isnull().sum()
            missing_df = pd.DataFrame(
                {
                    "Column": missing_data.index,
                    "Missing Count": missing_data.values,
                    "Percentage": (missing_data.values / len(st.session_state.df))
                    * 100,
                }
            )

            missing_df_filtered = missing_df[missing_df["Missing Count"] > 0]

            if not missing_df_filtered.empty:
                st.dataframe(missing_df_filtered, width="stretch")
            else:
                st.success("✅ No missing values found!")

        with col2:
            st.subheader("🔤 Categorical Columns")
            categorical_cols = st.session_state.df.select_dtypes(
                include=["object"]
            ).columns.tolist()

            if categorical_cols:
                for i, col in enumerate(categorical_cols, 1):
                    st.markdown(f"**{i}.** `{col}`")

                st.info(f"Total: {len(categorical_cols)} categorical columns")
            else:
                st.warning("No categorical columns found")

            st.subheader("🎯 Data Types")
            dtype_df = pd.DataFrame(
                {
                    "Column": st.session_state.df.dtypes.index,
                    "Data Type": st.session_state.df.dtypes.values.astype(str),
                }
            )
            st.dataframe(dtype_df, width="stretch")

        st.subheader("📈 Feature Analysis")
        selected_feature = st.selectbox(
            "Select a feature to analyze:", st.session_state.features
        )

        if selected_feature:
            col1, col2, col3 = st.columns(3)

            feature_data = st.session_state.df[selected_feature]

            with col1:
                st.metric(
                    "Mean",
                    f"{feature_data.mean():.2f}"
                    if feature_data.dtype in ["int64", "float64"]
                    else "N/A",
                )
            with col2:
                st.metric(
                    "Std Dev",
                    f"{feature_data.std():.2f}"
                    if feature_data.dtype in ["int64", "float64"]
                    else "N/A",
                )
            with col3:
                st.metric("Unique Values", feature_data.nunique())

            if feature_data.dtype in ["int64", "float64"]:
                fig = px.histogram(
                    x=feature_data, title=f"Distribution of {selected_feature}"
                )
                st.plotly_chart(fig, width="stretch")
            else:
                value_counts = feature_data.value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 10 Values in {selected_feature}",
                )
                st.plotly_chart(fig, width="stretch")
