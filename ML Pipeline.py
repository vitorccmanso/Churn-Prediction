import pandas as pd
import pickle
import warnings
import tkinter as tk
from tkinter import filedialog
warnings.filterwarnings("ignore")

class PredictPipeline:
    """
    A class for predicting client chrun using a pre-trained model and preprocessing pipeline.

    Methods:
    - __init__: Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
    - preprocess_data: Preprocesses the input data, including feature engineering and transformation
    - predict: Predicts client chrun
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
        """
        with open("./artifacts/preprocessor_reduced.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        with open("./artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def preprocess_data(self, input_data):
        """
        Preprocesses the input data, including feature engineering and transformation

        Parameters:
        - input_data (pandas.DataFrame): The input data to be processed

        Returns:
        - pandas.DataFrame: The processed input data
        """
        # Check if the uploaded dataset contains all required columns, group "NumOfProducts" and specify columns for transformations
        columns = ["Geography", "Gender", "Age", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember"]
        if not set(columns).issubset(input_data.columns):
            raise ValueError(f"Dataset must contain all of these columns: {columns}")
        input_data = input_data[columns]
        if input_data["NumOfProducts"].dtype not in ["object", "categorical", str]:
            input_data["NumOfProducts"] = input_data["NumOfProducts"].apply(lambda x: "3 or more" if x >= 3 else str(x))
        one_hot_cols = input_data.select_dtypes(include="object").columns
        # Apply preprocessor object to input_data
        input_data = self.preprocessor.transform(input_data)
        one_hot_features = list(self.preprocessor.named_transformers_["cat_onehot"]["onehot"].get_feature_names_out(one_hot_cols))
        feature_names = ["Balance", "Age", "HasCrCard", "IsActiveMember"] + one_hot_features
        return pd.DataFrame(input_data, columns=feature_names)

    def predict(self, data):
        """
        Predicts clients churn

        Parameters:
        - data: The input data for prediction
        """
        preds_proba = self.model.predict_proba(self.preprocess_data(data))[:,1]
        preds = (preds_proba >= 0.3).astype(int)
        pd.DataFrame({"rowNumber": data["RowNumber"], "predictedValues": preds}).to_csv("results.csv", index=False)
        print("Predictions saved to results.csv")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    dataset_path = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv"), ("Excel Files", "*.xlsx *.xls")]
    )
    if dataset_path:
        try:
            data = pd.read_csv(dataset_path, sep=None, engine="python")
            results = PredictPipeline().predict(data)           
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("No file was selected")