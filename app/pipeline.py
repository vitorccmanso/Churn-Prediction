import pandas as pd
import numpy as np
import pickle

class PredictPipeline:
    """
    A class for predicting client chrun using a pre-trained model and preprocessing pipeline.

    Methods:
    - __init__: Initializes the PredictPipeline object by loading the mappings, preprocessor, and model from .pkl files
    - preprocess_dataset: Processes the input dataset, ensuring it contains the required columns
    - preprocess_data: Preprocesses the input data, including feature engineering and transformation
    - get_feature_names: Retrieves the feature names after preprocessing is applied
    - predict: Predicts client chrun
    - results_df: Saves predictions to a CSV file and returns a list of dictionaries for HTML rendering
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
        """
        preprocessor_path = "app/artifacts/preprocessor.pkl"
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
        model_path = "app/artifacts/model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_data(self, input_data):
        """
        Processes the input dataset, ensuring it contains the required columns

        Parameters:
        - input_data (pandas.DataFrame): The input data to be processed

        Returns:
        - pandas.DataFrame: The processed input data
        """
        # Check if the uploaded dataset contains all required columns, group "NumOfProducts" and specify columns for transformations
        columns = ["Geography", "Gender", "Age", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember"]
        if not set(columns).issubset(input_data.columns):
            raise ValueError("Dataset must contain all the columns listed above")
        input_data = input_data[columns]
        if input_data["NumOfProducts"].dtype in [int, float]:
            input_data["NumOfProducts"] = input_data["NumOfProducts"].apply(lambda x: "3 or more" if x >= 3 else str(x))
        one_hot_cols = input_data.select_dtypes(include="object").columns

        # Apply preprocessor object to input_data
        input_data = self.preprocessor.transform(input_data)
        one_hot_features = list(self.preprocessor.named_transformers_["cat_onehot"]["onehot"].get_feature_names_out(one_hot_cols))
        feature_names = ["Balance"] + ["Age"] + ["HasCrCard", "IsActiveMember"] + one_hot_features
        new_data = pd.DataFrame(input_data, columns=feature_names)
        return new_data

    def predict(self, data, path, manual_data=False):
        """
        Predicts clients churn

        Parameters:
        - data: The input data for prediction

        Returns:
        - list: The predicted chances of a client defaulting
        """
        preds_proba = self.model.predict_proba(self.preprocess_data(data))[:,1]
        preds = (preds_proba >= 0.3).astype(int)
        if manual_data:
            prediction_classes = ["No Churn", "Churn"]
            predicted_class = prediction_classes[preds[0]]
            return predicted_class
        results_df = pd.DataFrame({
            "rowNumber": data["RowNumber"],
            "predictedValues": preds
        })
        # Save results to a temporary CSV file and convert DataFrame to a list of dictionaries for rendering in HTML
        results_df.to_csv(path, index=False)
        results = results_df.to_dict(orient="records")
        return results

class CustomData:
    """ 
    A class representing custom datasets for client information

    Attributes:
    - Geography (str): The clients country
    - Gender (str): The clients gender
    - Age (int): The clients age
    - Balance (float): Balance or value associated with the clients account
    - NumOfProducts (int): The number of products the client has
    - HasCrCard (int): If the client owns or not a credit card
    - IsActiveMember (int): If the client is an active member or not

    Methods:
    - __init__: Initializes the CustomData object with the provided attributes
    - get_data_as_dataframe: Converts the CustomData object into a pandas DataFrame
    """
    def __init__(self, Geography: str,
                    Gender: str,
                    Age: int,
                    Balance: float,
                    NumOfProducts: int,
                    HasCrCard: int,
                    IsActiveMember: int):
        """
        Initializes the CustomData object with the provided attributes
        """
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember

    def get_data_as_dataframe(self):
        """
        Converts the CustomData object into a pandas DataFrame

        Returns:
        - pd.DataFrame: The CustomData object as a DataFrame
        """
        custom_data_input_dict = {
            "Geography": [self.Geography],
            "Gender": [self.Gender],
            "Age": [self.Age],
            "Balance": [self.Balance],
            "NumOfProducts": [self.NumOfProducts],
            "HasCrCard": [self.HasCrCard],
            "IsActiveMember": [self.IsActiveMember],
        }
        return pd.DataFrame(custom_data_input_dict)