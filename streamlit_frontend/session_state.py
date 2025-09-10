import streamlit as st


SESSION_DEFAULTS = {
    "uploaded_file_path": None,
    "features": None,
    "label_column": None,
    "data_loaded": False,
    "df": None,
    "training_completed": False,
    "training_results": None,
}


def initialize_session_state():
    """Initialize expected session state variables with defaults."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
