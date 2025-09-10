import pandas as pd
import plotly.express as px
import streamlit as st


def display_visualizations(api_client):
    """Display visualizations page"""
    st.header("📈 Data Visualizations")

    if st.session_state.features:
        st.subheader("🎛️ Feature Selection")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Available Features:** {', '.join(st.session_state.features)}")
        with col2:
            st.info(f"**Label Column:** {st.session_state.label_column}")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "📊 Scatter Plot",
                "📈 Histogram",
                "📉 Line Plot",
                "🔥 Correlation Matrix",
                "📦 Box Plot",
                "🔍 Pair Plot",
                "📈 Area Plot",
            ]
        )

        with tab1:
            display_scatter_plot(api_client)

        with tab2:
            display_histogram_plot(api_client)

        with tab3:
            display_line_plot(api_client)

        with tab4:
            display_correlation_matrix(api_client)

        with tab5:
            display_box_plot(api_client)

        with tab6:
            display_pair_plot(api_client)

        with tab7:
            display_area_plot(api_client)


def display_scatter_plot(api_client):
    st.subheader("📊 Scatter Plot")

    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox(
            "Select X-axis feature:", st.session_state.features, key="scatter_x"
        )
    with col2:
        feature2 = st.selectbox(
            "Select Y-axis feature:", st.session_state.features, key="scatter_y"
        )

    if st.button("Generate Scatter Plot", key="scatter_btn"):
        with st.spinner("Generating scatter plot..."):
            scatter_data = api_client.get_scatter_plot(
                st.session_state.uploaded_file_path, feature1, feature2
            )
            if scatter_data:
                try:
                    fig = px.scatter(
                        x=scatter_data[feature1],
                        y=scatter_data[feature2],
                        labels={feature1: feature1, feature2: feature2},
                        title=f"Scatter Plot: {feature1} vs {feature2}",
                    )
                    fig.update_layout(width=800, height=500)
                    st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
                    st.write("Raw data:", scatter_data)


def display_histogram_plot(api_client):
    st.subheader("📈 Histogram")

    if st.button("Generate Histogram", key="hist_btn"):
        with st.spinner("Generating histogram..."):
            hist_data = api_client.get_histogram_plot(
                st.session_state.uploaded_file_path
            )
            if hist_data:
                try:
                    df = pd.DataFrame(hist_data)

                    numerical_cols = df.select_dtypes(include=["number"]).columns

                    if len(numerical_cols) > 0:
                        for col in numerical_cols:
                            fig = px.histogram(
                                df, x=col, title=f"Histogram of {col}", nbins=20
                            )
                            fig.update_layout(width=800, height=400)
                            st.plotly_chart(fig, width="stretch")
                    else:
                        st.warning("No numerical columns found for histogram.")

                except Exception as e:
                    st.error(f"Error creating histogram: {str(e)}")
                    st.write("Raw data:", hist_data)


def display_line_plot(api_client):
    st.subheader("📉 Line Plot")

    if st.button("Generate Line Plot", key="line_btn"):
        with st.spinner("Generating line plot..."):
            line_data = api_client.get_line_plot(st.session_state.uploaded_file_path)
            if line_data:
                try:
                    df = pd.DataFrame(line_data)

                    numerical_cols = df.select_dtypes(include=["number"]).columns

                    if len(numerical_cols) > 0:
                        fig = px.line(
                            df.reset_index(),
                            x="index",
                            y=numerical_cols.tolist(),
                            title="Line Plot of Numerical Features",
                        )
                        fig.update_layout(
                            width=800,
                            height=500,
                            xaxis_title="Data Point Index",
                            yaxis_title="Values",
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.warning("No numerical columns found for line plot.")

                except Exception as e:
                    st.error(f"Error creating line plot: {str(e)}")
                    st.write("Raw data:", line_data)


def display_correlation_matrix(api_client):
    st.subheader("🔥 Correlation Matrix")

    if st.button("Generate Correlation Matrix", key="corr_btn"):
        with st.spinner("Generating correlation matrix..."):
            corr_data = api_client.get_correlation_matrix(
                st.session_state.uploaded_file_path
            )
            if corr_data:
                try:
                    df = pd.DataFrame(corr_data)

                    if not df.empty:
                        fig = px.imshow(
                            df.values,
                            labels=dict(
                                x="Features", y="Features", color="Correlation"
                            ),
                            x=df.columns,
                            y=df.index,
                            color_continuous_scale="RdBu",
                            aspect="auto",
                            title="Correlation Matrix Heatmap",
                        )
                        fig.update_layout(width=800, height=600)
                        st.plotly_chart(fig, width="stretch")

                        st.subheader("Correlation Values")
                        st.dataframe(df.round(3))
                    else:
                        st.warning("No correlation data available.")

                except Exception as e:
                    st.error(f"Error creating correlation matrix: {str(e)}")
                    st.write("Raw data:", corr_data)


def display_box_plot(api_client):
    st.subheader("📦 Box Plot")

    feature = st.selectbox(
        "Select feature for box plot:", st.session_state.features, key="box_feature"
    )

    if st.button("Generate Box Plot", key="box_btn"):
        with st.spinner("Generating box plot..."):
            box_data = api_client.get_box_plot(
                st.session_state.uploaded_file_path, feature
            )
            if box_data:
                try:
                    feature_data = box_data.get(feature, [])

                    if feature_data:
                        fig = px.box(y=feature_data, title=f"Box Plot: {feature}")
                        fig.update_layout(width=600, height=500, yaxis_title=feature)
                        st.plotly_chart(fig, width="stretch")

                        series = pd.Series(feature_data)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{series.mean():.2f}")
                        with col2:
                            st.metric("Median", f"{series.median():.2f}")
                        with col3:
                            st.metric("Std Dev", f"{series.std():.2f}")
                        with col4:
                            st.metric(
                                "IQR",
                                f"{series.quantile(0.75) - series.quantile(0.25):.2f}",
                            )
                    else:
                        st.warning(f"No data available for feature: {feature}")

                except Exception as e:
                    st.error(f"Error creating box plot: {str(e)}")
                    st.write("Raw data:", box_data)


def display_pair_plot(api_client):
    st.subheader("🔍 Pair Plot")

    st.info("This plot shows pairwise relationships between all numerical features.")

    if st.button("Generate Pair Plot", key="pair_btn"):
        with st.spinner("Generating pair plot... (this may take a while)"):
            pair_data = api_client.get_pair_plot(st.session_state.uploaded_file_path)
            if pair_data:
                try:
                    df = pd.DataFrame(pair_data)

                    numerical_cols = df.select_dtypes(
                        include=["number"]
                    ).columns.tolist()

                    if len(numerical_cols) >= 2:
                        fig = px.scatter_matrix(
                            df,
                            dimensions=numerical_cols,
                            title="Pair Plot - Scatter Matrix of Numerical Features",
                        )
                        fig.update_layout(width=800, height=800)
                        st.plotly_chart(fig, width="stretch")

                        st.info(
                            f"Showing pairwise relationships for {len(numerical_cols)} numerical features: {', '.join(numerical_cols)}"
                        )
                    else:
                        st.warning("Need at least 2 numerical columns for pair plot.")

                except Exception as e:
                    st.error(f"Error creating pair plot: {str(e)}")
                    st.write(
                        "Raw data sample:",
                        str(pair_data)[:500] + "..."
                        if len(str(pair_data)) > 500
                        else str(pair_data),
                    )


def display_area_plot(api_client):
    st.subheader("📈 Area Plot")

    feature = st.selectbox(
        "Select feature for area plot:", st.session_state.features, key="area_feature"
    )

    if st.button("Generate Area Plot", key="area_btn"):
        with st.spinner("Generating area plot..."):
            area_data = api_client.get_area_plot(
                st.session_state.uploaded_file_path, feature
            )
            if area_data:
                try:
                    feature_data = area_data.get(feature, [])

                    if feature_data:
                        df = pd.DataFrame(
                            {"index": range(len(feature_data)), feature: feature_data}
                        )

                        fig = px.area(
                            df, x="index", y=feature, title=f"Area Plot: {feature}"
                        )
                        fig.update_layout(
                            width=800,
                            height=500,
                            xaxis_title="Data Point Index",
                            yaxis_title=feature,
                        )
                        st.plotly_chart(fig, width="stretch")

                        cumsum = pd.Series(feature_data).cumsum()
                        st.subheader("Cumulative Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Sum", f"{cumsum.iloc[-1]:.2f}")
                        with col2:
                            st.metric("Data Points", len(feature_data))

                    else:
                        st.warning(f"No data available for feature: {feature}")

                except Exception as e:
                    st.error(f"Error creating area plot: {str(e)}")
                    st.write("Raw data:", area_data)
