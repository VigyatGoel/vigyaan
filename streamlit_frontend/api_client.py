import requests
import streamlit as st


class MLPlatformAPI:
    """API client for the ML Platform"""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def upload_csv(self, file):
        """Upload CSV file to the API"""
        try:
            files = {"file": (file.name, file, "text/csv")}
            response = requests.post(
                f"{self.base_url}/csv_file/upload_csv", files=files
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading file: {str(e)}")
            return None

    def extract_features(self, file_path: str):
        """Extract features from CSV file"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/csv_file/extract_features", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error extracting features: {str(e)}")
            return None

    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            params = {"file_path": file_path}
            response = requests.delete(
                f"{self.base_url}/csv_file/cleanup_temp_file", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def get_data_info(self, file_path: str):
        """Get data info"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/data_summary/data_info", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting data info: {str(e)}")
            return None

    def get_data_description(self, file_path: str):
        """Get data description"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/data_summary/data_description", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting data description: {str(e)}")
            return None

    def get_file_info(self, file_path: str):
        """Get file info"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/data_summary/file_info", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting file info: {str(e)}")
            return None

    def get_scatter_plot(self, file_path: str, feature1: str, feature2: str):
        """Get scatter plot data"""
        try:
            params = {"csv_file": file_path, "feature1": feature1, "feature2": feature2}
            response = requests.get(
                f"{self.base_url}/data_science/scatter_plot", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting scatter plot: {str(e)}")
            return None

    def get_histogram_plot(self, file_path: str):
        """Get histogram plot data"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/data_science/histogram_plot", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting histogram plot: {str(e)}")
            return None

    def get_line_plot(self, file_path: str):
        """Get line plot data"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/data_science/line_plot", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting line plot: {str(e)}")
            return None

    def get_correlation_matrix(self, file_path: str):
        """Get correlation matrix data"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/data_science/correlation_matrix", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting correlation matrix: {str(e)}")
            return None

    def get_box_plot(self, file_path: str, feature1: str):
        """Get box plot data"""
        try:
            params = {"csv_file": file_path, "feature1": feature1}
            response = requests.get(
                f"{self.base_url}/data_science/box_plot", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting box plot: {str(e)}")
            return None

    def get_pair_plot(self, file_path: str):
        """Get pair plot data"""
        try:
            params = {"csv_file": file_path}
            response = requests.get(
                f"{self.base_url}/data_science/pair_plot", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting pair plot: {str(e)}")
            return None

    def get_area_plot(self, file_path: str, feature1: str):
        """Get area plot data"""
        try:
            params = {"csv_file": file_path, "feature1": feature1}
            response = requests.get(
                f"{self.base_url}/data_science/area_plot", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting area plot: {str(e)}")
            return None

    def train_models(self, file_path: str, target_var: str):
        """Train machine learning models"""
        try:
            params = {"csv_file": file_path, "target_var": target_var}
            response = requests.get(
                f"{self.base_url}/machine_learning/train", params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error training models: {str(e)}")
            return None

    def download_model(self, model_name: str):
        """Download a trained model or preprocessing artifact"""
        try:
            params = {"model_name": model_name}
            response = requests.get(
                f"{self.base_url}/machine_learning/download", params=params
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {str(e)}")
            return None
