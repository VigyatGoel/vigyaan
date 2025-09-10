import pandas as pd
import plotly.express as px
import streamlit as st


def display_machine_learning(api_client):
    """Display machine learning page"""
    st.header("🤖 Machine Learning")

    if st.session_state.df is None:
        st.error("No data loaded. Please upload a CSV file first.")
        return

    st.subheader("1. Select Target Variable")
    target_columns = st.session_state.df.columns.tolist()
    selected_target = st.selectbox(
        "Choose the target variable for prediction:",
        target_columns,
        index=len(target_columns) - 1 if target_columns else 0,
        key="target_variable_select",
    )

    st.subheader("2. Data Preprocessing")
    st.info("The system will automatically handle:")
    st.markdown(
        """
    - Missing values (filled with mean for numerical, mode for categorical)
    - Categorical encoding (Label encoding)
    - Feature scaling (StandardScaler)
    """
    )

    def render_training_results(models_data):
        st.subheader("3. Model Performance Metrics")
        metrics_df = pd.DataFrame(models_data).T.round(4)
        st.dataframe(metrics_df, width="stretch")
        col1, col2 = st.columns(2)
        with col1:
            accuracy_fig = px.bar(
                x=metrics_df.index,
                y=metrics_df["accuracy"],
                title="Model Accuracy Comparison",
                labels={"x": "Model", "y": "Accuracy"},
            )
            st.plotly_chart(accuracy_fig, width="stretch", key="accuracy_chart")
        with col2:
            f1_fig = px.bar(
                x=metrics_df.index,
                y=metrics_df["f1_score"],
                title="F1 Score Comparison",
                labels={"x": "Model", "y": "F1 Score"},
            )
            st.plotly_chart(f1_fig, width="stretch", key="f1_chart")

        st.subheader("4. Download Trained Models")
        for model_name, metrics in models_data.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{model_name}**")
                st.write(
                    f"Accuracy: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1_score']:.4f}"
                )
            with col2:
                model_bytes_key = f"model_bytes_{model_name}"
                if model_bytes_key not in st.session_state:
                    st.session_state[model_bytes_key] = api_client.download_model(
                        f"{model_name}.pkl"
                    )
                if st.session_state[model_bytes_key]:
                    st.download_button(
                        label="⬇️ Model",
                        data=st.session_state[model_bytes_key],
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream",
                        key=f"download_{model_name}",
                    )
            with col3:
                st.caption("Template code below ⬇️")

        st.subheader("5. Using Your Models")
        st.markdown(
            """
        **Quick Steps:**
        1. Download a model (above) and the preprocessing files (below).
        2. Create a simple loader script (example):
        ```python
        import os
        import numpy as np
        import pickle


        def predict(features: dict, model_path: str):
            d = os.path.dirname(model_path) or "."
            model = pickle.load(open(model_path, "rb"))
            scaler = pickle.load(open(os.path.join(d, "scaler.pkl"), "rb"))
            imputer = pickle.load(open(os.path.join(d, "imputer.pkl"), "rb"))

            try:
                names = pickle.load(open(os.path.join(d, "feature_names.pkl"), "rb"))
            except Exception:
                names = list(features.keys())

            ordered = [features.get(n) for n in names]

            X = np.array(ordered).reshape(1, -1)
            X = scaler.transform(imputer.transform(X))

            pred = model.predict(X)[0]
            probs = (
                model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
            )

            return {"prediction": pred, "probabilities": probs}


        # usage:
        result = predict(
            {"feature1": 5.1, "feature2": 3.5, "feature3": 1.4}, "./LogisticRegression.pkl"
        )
        print(result)

        ```
        3. Run predictions in your own scripts.
        """
        )
        st.markdown("**Dependencies (install once):**")
        st.code("pip install numpy scikit-learn", language="bash")
        st.caption("Optional (for data wrangling / inspection): pip install pandas")

        st.subheader("6. Download Preprocessing Files")
        col1, col2, col3 = st.columns(3)
        preprocessing_files = ["scaler.pkl", "imputer.pkl", "feature_names.pkl"]
        for i, file_name in enumerate(preprocessing_files):
            with [col1, col2, col3][i]:
                prep_key = f"prep_bytes_{file_name}"
                if prep_key not in st.session_state:
                    st.session_state[prep_key] = api_client.download_model(file_name)
                if st.session_state[prep_key]:
                    st.download_button(
                        label=f"⬇️ {file_name}",
                        data=st.session_state[prep_key],
                        file_name=file_name,
                        mime="application/octet-stream",
                        key=f"prep_{file_name}",
                    )

    if st.session_state.get("training_completed") and st.session_state.get(
        "training_results"
    ):
        # st.info(
        #     f"Showing previously trained models for target: {st.session_state.get('selected_target', '(unknown)')}"
        # )
        render_training_results(st.session_state.training_results)

    if not st.session_state.get("training_completed"):
        if st.button("🚀 Train Models", type="primary"):
            if not selected_target:
                st.error("Please select a target variable")
                return
            with st.spinner("Training models... This may take a few moments."):
                training_result = api_client.train_models(
                    st.session_state.uploaded_file_path, selected_target
                )
                if training_result and "models" in training_result:
                    st.session_state.training_completed = True
                    st.session_state.training_results = training_result["models"]
                    st.session_state.selected_target = selected_target
                    st.success("✅ Models trained successfully!")
                    st.rerun()
                else:
                    st.error(
                        "Failed to train models. Please check your data and try again."
                    )
