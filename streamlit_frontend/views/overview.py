import pandas as pd
import streamlit as st


def display_data_overview(api_client):
    """Display data overview page"""
    st.header("📊 Data Overview")

    if st.session_state.df is not None:
        st.subheader("📋 Data Preview")
        st.dataframe(st.session_state.df.head(10), width="stretch")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.df.shape[1])
        with col3:
            st.metric(
                "Memory Usage",
                f"{st.session_state.df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            )

    st.subheader("📄 File Information")
    with st.spinner("Loading file info..."):
        file_info = api_client.get_file_info(st.session_state.uploaded_file_path)
        if file_info:
            col1, col2 = st.columns(2)
            with col1:
                display_name = st.session_state.get(
                    "original_filename", file_info.get("file_name", "N/A")
                )
                st.metric("File Name", display_name)
            with col2:
                file_size = file_info.get("file_size_MB", 0)
                st.metric("File Size", f"{file_size:.4f} MB")

    st.subheader("ℹ️ Data Information")
    with st.spinner("Loading data info..."):
        data_info = api_client.get_data_info(st.session_state.uploaded_file_path)
        if data_info:
            info_text = data_info.get("data_info", "")
            if info_text:
                lines = info_text.split("\n")

                st.markdown("**Dataset Overview:**")
                for line in lines:
                    if (
                        "entries" in line
                        or "columns" in line
                        or "dtypes" in line
                        or "memory usage" in line
                    ):
                        st.text(line.strip())

                st.markdown("**Column Details:**")

                column_data = []
                capture_columns = False

                for line in lines:
                    line = line.strip()
                    if line.startswith("#") and "Column" in line:
                        capture_columns = True
                        continue
                    elif line.startswith("dtypes:"):
                        capture_columns = False
                        continue
                    elif capture_columns and line and not line.startswith("---"):
                        parts = line.split()
                        if len(parts) >= 4:
                            col_num = parts[0]
                            col_name = parts[1]
                            non_null_count = " ".join(parts[2:4])
                            dtype = parts[4] if len(parts) > 4 else "N/A"
                            column_data.append(
                                {
                                    "#": col_num,
                                    "Column": col_name,
                                    "Non-Null Count": non_null_count,
                                    "Data Type": dtype,
                                }
                            )

                if column_data:
                    df_info = pd.DataFrame(column_data)
                    st.dataframe(df_info, width="stretch")
                else:
                    st.text(info_text)

    st.subheader("📊 Statistical Description")
    with st.spinner("Loading data description..."):
        data_description = api_client.get_data_description(
            st.session_state.uploaded_file_path
        )
        if data_description:
            try:
                desc_df = pd.DataFrame(data_description)

                if not desc_df.empty:
                    st.dataframe(desc_df.round(3), width="stretch")

                    if len(desc_df.columns) > 0:
                        st.markdown("**Key Statistics:**")

                        numeric_cols = desc_df.select_dtypes(include=["number"]).columns

                        if len(numeric_cols) >= 2:
                            col1, col2, col3, col4 = st.columns(4)

                            first_col = numeric_cols[0]
                            if "mean" in desc_df.index:
                                with col1:
                                    st.metric(
                                        f"Mean ({first_col})",
                                        f"{desc_df.loc['mean', first_col]:.3f}",
                                    )
                            if "std" in desc_df.index:
                                with col2:
                                    st.metric(
                                        f"Std Dev ({first_col})",
                                        f"{desc_df.loc['std', first_col]:.3f}",
                                    )
                            if "min" in desc_df.index:
                                with col3:
                                    st.metric(
                                        f"Min ({first_col})",
                                        f"{desc_df.loc['min', first_col]:.3f}",
                                    )
                            if "max" in desc_df.index:
                                with col4:
                                    st.metric(
                                        f"Max ({first_col})",
                                        f"{desc_df.loc['max', first_col]:.3f}",
                                    )
                else:
                    st.warning("No statistical description data available.")

            except Exception as e:
                st.error(f"Error formatting data description: {str(e)}")
                st.json(data_description)
