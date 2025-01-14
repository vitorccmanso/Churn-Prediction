import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
import pickle

mlflow_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
uri = os.environ.get("uri")

class DataPreprocess:
    """
    A class for preprocessing data, including feature engineering, transformation, and splitting data into train-test sets

    Methods:
    - __init__: Initializes the DataPreprocess object
    - save_preprocessor: Saves the preprocessor object to a file
    - load_preprocessor: Loads the preprocessor object from a file
    - get_feature_names: Retrieves feature names after applying transformations in the preprocessor pipeline
    - preprocessor: Creates and returns a preprocessor pipeline for data preprocessing
    - preprocess_data: Preprocesses data for training, applying transformations, feature engineering, and splitting into train-test sets
    """
    def __init__(self):
        pass

    def save_preprocessor(self, preprocessor, reduced_columns):
        """
        Saves the preprocessor object to a file

        Parameters:
        - preprocessor (sklearn.pipeline.Pipeline): The preprocessor object to be saved
        - reduced_columns (bool): Flag indicating if the input data contains all columns or not
        """
        if not os.path.exists("../artifacts"):
            os.makedirs("../artifacts")
        if not reduced_columns:
            with open("../artifacts/preprocessor_all.pkl", "wb") as f_all:
                pickle.dump(preprocessor, f_all)
        else:
            with open("../artifacts/preprocessor_reduced.pkl", "wb") as f_all:
                pickle.dump(preprocessor, f_all)
    
    def load_preprocessor(self, preprocessor):
        """
        Loads the preprocessor object from a file

        Parameters:
        - preprocessor (str): The name of the preprocessor to be loaded

        Returns:
        - preprocessor: The loaded preprocessor object
        """
        with open(f"../artifacts/{preprocessor}.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return preprocessor

    def get_feature_names(self, preprocessor, num_only_scale_cols, log_cols, cat_cols, one_hot_cols):
        """
        Retrieves the feature names after preprocessing is applied

        Parameters:
        - preprocessor (sklearn.pipeline.Pipeline): The preprocessor object used for transformations
        - num_only_scale_cols (list): List of numeric columns without transformations
        - log_cols (list): List of column names for which log transformation is applied
        - cat_cols (list): List of categorical columns that don't need any kind of transformation
        - one_hot_cols (list): List of categorical columns for which OneHotEncoding is applied

        Returns:
        - feature_names (list): List of feature names after preprocessing
        """
        numeric_features = num_only_scale_cols + log_cols
        categorical_features = cat_cols
        one_hot_features = list(preprocessor.named_transformers_["cat_onehot"]["onehot"].get_feature_names_out(one_hot_cols))
        feature_names = numeric_features + categorical_features + one_hot_features
        return feature_names

    def preprocessor(self, num_only_scale_cols, log_cols, cat_cols, one_hot_cols):
        """
        Creates and returns a preprocessor pipeline for data preprocessing

        Parameters:
        - num_only_scale_cols (list): List of numeric columns without transformations
        - log_cols (list): List of column names for which log transformation is applied
        - cat_cols (list): List of categorical columns that don't need any kind of transformation
        - one_hot_cols (list): List of categorical columns for which OneHotEncoding is applied

        Returns:
        - preprocessor (sklearn.compose.ColumnTransformer): Preprocessor pipeline for data preprocessing
        """
        log_transformer = Pipeline(steps=[
            ("log_transformation", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", MinMaxScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder())
        ])
        transformers=[
            ("num_only_scale", MinMaxScaler(), num_only_scale_cols),
            ("num_log", log_transformer, log_cols),
            ("cat_no_transform", FunctionTransformer(validate=True), cat_cols),
            ("cat_onehot", categorical_transformer, one_hot_cols)
        ]

        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False, remainder="drop")
        return preprocessor

    def preprocess_data(self, data, target_name=None, test_size=None, reduced_columns=False, test_data=False, preprocessor=None):
        """
        Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets

        Parameters:
        - data (pandas.DataFrame): The input DataFrame containing raw data
        - target_name (str): The name of the target variable to predict
        - test_size (float): Proportion of the dataset to include in the test split
        - reduced_columns (bool): Flag indicating if the input data contains all columns or not
        - test_data (bool): Flag indicating if the input data is test data. If true, the function returns only the test dataframe
        - preprocessor (str): The name of the preprocessor to be loaded

        Returns:
        - X (pandas.Dataframe): Preprocessed test data for predictions
        - X_train (pandas.DataFrame): Preprocessed features for training set
        - X_test (pandas.DataFrame): Preprocessed features for testing set
        - y_train (pandas.Series): Target labels for training set
        - y_test (pandas.Series): Target labels for testing set
        """
        # Drop unnecessary columns, group "NumOfProducts" if test data and specify columns for transformations
        data_process = data.drop(columns=["RowNumber", "CustomerId", "Surname", target_name], errors='ignore')
        data_process["NumOfProducts"] = data_process["NumOfProducts"].apply(lambda x: "3 or more" if x >= 3 else str(x)) if test_data else data_process["NumOfProducts"]
        cat_cols = ["HasCrCard", "IsActiveMember"]
        num_only_scale_cols = ["CreditScore", "Tenure", "Balance", "EstimatedSalary"] if not reduced_columns else ["Balance"]
        log_cols = ["Age"]
        one_hot_cols = data_process.select_dtypes(include="object").columns

        if test_data:
            preprocessor = self.load_preprocessor(preprocessor)
            X = preprocessor.transform(data_process)
            feature_names = self.get_feature_names(preprocessor, num_only_scale_cols, log_cols, cat_cols, one_hot_cols)
            X = pd.DataFrame(X, columns=feature_names)
            return X

        preprocessor = self.preprocessor(num_only_scale_cols, log_cols, cat_cols, one_hot_cols)
        data_preprocessed = preprocessor.fit_transform(data_process)
        feature_names = self.get_feature_names(preprocessor, num_only_scale_cols, log_cols, cat_cols, one_hot_cols)
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=feature_names)

        X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data[target_name], test_size=test_size, stratify=data[target_name], shuffle=True, random_state=42)
        self.save_preprocessor(preprocessor, reduced_columns)

        return X_train, X_test, y_train, y_test

class ModelTraining:
    """
    A class for training machine learning models, evaluating their performance, and saving the best one

    Methods:
    - __init__: Initializes the ModelTraining object
    - save_model: Saves the specified model to a pkl file
    - initiate_model_trainer: Initiates the model training process and evaluates multiple models
    - evaluate_models: Evaluates multiple models using random search cross-validation and logs the results with MLflow
    """
    def __init__(self):
        pass

    def save_model(self, model_name, version, save_folder, save_filename):
        """
        Save the specified model to a pkl file

        Parameters:
        - model_name (str): The name of the model to save
        - version (int): The version of the model to save
        - save_folder (str): The folder path where the model will be saved
        - save_filename (str): The filename for the pkl file
        """
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)

        # Get the correct version of the registered model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        model_versions_sorted = sorted(model_versions, key=lambda v: int(v.version))
        requested_version = model_versions_sorted[version - 1]
        
        # Construct the logged model path
        run_id = requested_version.run_id
        artifact_path = requested_version.source.split("/")[-1]
        logged_model = f"runs:/{run_id}/{artifact_path}"

        # Load the model from MLflow and saves it to a pkl file
        loaded_model = mlflow.sklearn.load_model(logged_model)
        file_path = os.path.join(save_folder, f"{save_filename}.pkl")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(loaded_model, f)

    def initiate_model_trainer(self, train_test, experiment_name, scoring, refit, use_smote=False):
        """
        Initiates the model training process

        Parameters:
        - train_test (tuple): A tuple containing the train-test split data in the format (X_train, y_train, X_test, y_test)
        - experiment_name (str): Name of the MLflow experiment where the results will be logged
        - scoring (list): The scoring metrics used for evaluating the models
        - refit (str): The metric to refit the best model
        - use_smote (bool): A boolean indicating whether to apply SMOTE for balancing the classes. Default is False

        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_tracking_uri(uri)
        X_train, y_train, X_test, y_test = train_test
        
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        
        params = {
            "Logistic Regression": {
                "solver": ["liblinear", "lbfgs"],
                "penalty":["l2", "l1", "elasticnet", None], 
                "C":[1.5, 1, 0.5, 0.1]
            },
            "Random Forest":{
                "criterion":["gini", "entropy", "log_loss"],
                "max_features":["sqrt", "log2"],
                "n_estimators": [25, 50, 100, 150, 200],
                "max_depth": [2, 5, 10, 20]
            },
            "XGBoost":{
                "n_estimators": [25, 50, 100, 150, 200],
                "max_depth": [2, 5, 10, 20],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
            },
            "SVM":{
                "C":[1.5, 1, 0.5, 0.1],
                "gamma": ["scale", "auto"],
                "tol": [0.01, 0.01, 0.1]
            }
        }
        
        model_report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, params=params, experiment_name=experiment_name, 
                                           scoring=scoring, refit=refit, use_smote=use_smote)
        
        return model_report

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params, experiment_name, scoring, refit, use_smote):
        """
        Evaluates multiple models using random search cross-validation and logs the results with MLflow

        Parameters:
        - X_train (array-like): Features of the training data
        - y_train (array-like): Target labels of the training data
        - X_test (array-like): Features of the testing data
        - y_test (array-like): Target labels of the testing data
        - models (dict): A dictionary containing the models to be evaluated
        - params (dict): A dictionary containing the hyperparameter grids for each model
        - experiment_name (str): Name of the MLflow experiment where the results will be logged
        - scoring (list): The scoring metrics used for evaluating the models
        - refit (str): The metric to refit the best model
        - use_smote (bool): A boolean indicating whether to apply SMOTE for balancing the classes

        Returns:
        - dict: A dictionary containing the evaluation report for each model.
        """
        mlflow.set_experiment(experiment_name)
        report = {}
        if use_smote:
            # Apply SMOTE only to the training data
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                param = params[model_name]
                param["class_weight"] = [None] if use_smote else ["balanced"]

                rs = RandomizedSearchCV(model, param, cv=5, scoring=scoring, refit=refit, random_state=42)
                search_result = rs.fit(X_train, y_train)
                model = search_result.best_estimator_
                y_pred = model.predict(X_test)
                mlflow.set_tags({"model_type": f"{model_name}-{experiment_name}", "smote_applied": use_smote})

                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred)
                roc = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                
                # Log metrics to MLflow
                mlflow.log_params(search_result.best_params_)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc_score", roc_auc)
                mlflow.log_metric("recall_score", recall)
                mlflow.log_metric("precision_score", precision)
                mlflow.sklearn.log_model(model, model_name, registered_model_name=f"{model_name} - {experiment_name}")
                
                # Store the model for visualization
                report[model_name] = {"model": model, "y_pred": y_pred, "roc_auc_score": roc_auc, "roc_curve": roc}      
        return report


class MetricsVisualizations:
    """
    A class for visualizing model evaluation metrics and results

    Attributes:
    - models (dict): A dictionary containing the trained models, with metrics and predictions for each model

    Methods:
    - __init__: Initializes the MetricsVisualizations object with a dictionary of models
    - create_subplots: Creates a figure and subplots with common settings
    - visualize_roc_curves: Visualizes ROC curves for each model, showing the trade-off between true positive and false positive rates
    - visualize_confusion_matrix: Visualizes confusion matrices for each model, displaying absolute and relative values
    - plot_precision_recall_threshold: Plots precision and recall against thresholds for each model, providing insight into performance at different classification thresholds
    - plot_feature_importance: Plots feature importance for each model using permutation importance
    """
    def __init__(self, models):
        """
        Initializes the MetricsVisualizations object with a dictionary of models

        Parameters:
        - models (dict): A dictionary containing the trained models, with metrics and predictions for each model
        """
        self.models = models

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - figsize (tuple): Figure size. Default is (18, 12)
        
        Returns:
        - fig (matplotlib.figure.Figure): The figure object
        - ax (numpy.ndarray): Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def visualize_roc_curves(self):
        """
        Visualizes ROC curves for each model
        """
        plt.figure(figsize=(12, 6))
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")

        for model_name, model_data in self.models.items():
            model_roc_auc = model_data["roc_auc_score"]
            fpr, tpr, thresholds = model_data["roc_curve"]
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_roc_auc:.3f})")
        plt.legend()
        plt.show()

    def visualize_confusion_matrix(self, y_test, rows, columns):
        """
        Visualizes confusion matrices for each model

        Parameters:
        - y_test (array-like): True labels of the test data.
        - rows (int): Number of rows for subplots
        - columns (int): Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(14, 10))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            y_pred = model_data["y_pred"]
            matrix = confusion_matrix(y_test, y_pred)

            # Plot the first heatmap
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax[i * 2])
            ax[i * 2].set_title(f"Confusion Matrix: {model_name} - Absolute Values")
            ax[i * 2].set_xlabel("Predicted Values")
            ax[i * 2].set_ylabel("Observed values")

            # Plot the second heatmap
            sns.heatmap(matrix / np.sum(matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[i * 2 + 1])
            ax[i * 2 + 1].set_title(f"Relative Values")
            ax[i * 2 + 1].set_xlabel("Predicted Values")
            ax[i * 2 + 1].set_ylabel("Observed values")

        fig.tight_layout()
        plt.show()

    def plot_precision_recall_threshold(self, y_test, X_test, rows, columns):
        """
        Plots precision, recall and f1-score vs thresholds for each model

        Parameters:
        - y_test (array-like): True labels of the test data
        - X_test (pandas.DataFrame): Features of the test data
        - rows (int): Number of rows for subplots
        - columns (int): Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(16, 10))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            y_pred_prob = model_data["model"].predict_proba(X_test)[:,1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])

            # Plot Precision-Recall vs Thresholds for each model
            ax[i].set_title(f"Precision X Recall vs Thresholds - {model_name}")
            ax[i].plot(thresholds, precisions[:-1], "b--", label="Precision")
            ax[i].plot(thresholds, recalls[:-1], "g-", label="Recall")
            ax[i].plot(thresholds, f1_scores, "r-", label="F1 Score")
            ax[i].plot([0.5, 0.5], [0, 1], "k--", label="0.5 Threshold")
            ax[i].set_ylabel("Score")
            ax[i].set_xlabel("Threshold")
            ax[i].legend(loc="center left")

            # Annotate metrics at 0.5 threshold
            y_pred = model_data["y_pred"]
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
            ax[i].plot(0.5, precision, "or")
            ax[i].annotate(f"{precision:.3f} precision", (0.51, precision))
            ax[i].plot(0.5, recall, "or")
            ax[i].annotate(f"{recall:.3f} recall", (0.51, recall))
            ax[i].plot(0.5, f1_score, "or")
            ax[i].annotate(f"{f1_score:.3f} F1 Score", (0.51, f1_score))

        fig.tight_layout()
        plt.show()

    def plot_feature_importance(self, y_test, X_test, metric, rows, columns):
        """
        Plots feature importance for each model using permutation importance

        Parameters:
        - y_test (array-like): True labels of the test data
        - X_test (DataFrame): Features of the test data, where each column represents a feature
        - metric (str): The scoring metric used for evaluating feature importance (e.g., "accuracy", "f1", etc.)
        - rows (int): Number of rows for the subplot grid
        - columns (int): Number of columns for the subplot grid
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(16, 10))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate and sort permutation importances
            result = permutation_importance(model_data["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=metric)
            sorted_importances_idx = result["importances_mean"].argsort()
            importances = pd.DataFrame(result.importances[sorted_importances_idx].T, columns=X_test.columns[sorted_importances_idx])

            # Plot boxplot of feature importances
            box = importances.plot.box(vert=False, whis=10, ax=ax[i])
            box.set_title(f"Feature Importance - {model_name}")
            box.axvline(x=0, color="k", linestyle="--")
            box.set_xlabel(f"Decay in {metric}")
            box.figure.tight_layout()

        fig.tight_layout()
        plt.show()

def compare_confusion_matrices(y_test, X_test, model_dicts_list, thresholds, rows, columns):
    """
    Compare confusion matrices for multiple models and thresholds

    Parameters:
    - y_test (array-like): True labels of the test data
    - X_test (DataFrame): Features of the test data, where each column represents a feature
    - model_dicts_list (list of tuples): List containing tuples with the following structure:
        - model_dict (dict): A dictionary where keys are model names and values contain model data
        - type_of_balancing (str): The type of balancing technique used to train the model (e.g., "Smote")
    - thresholds (dict): A dictionary mapping model identifiers (constructed from model name, balancing, and metric) to their corresponding thresholds.
    - rows (int): Number of rows for the subplot grid
    - columns (int): Number of columns for the subplot grid
    """
    fig, ax = MetricsVisualizations(model_dicts_list).create_subplots(rows, columns, figsize=(12, 4))
    plot_index = 0

    # Track which models have already been processed
    processed_models = set()

    # Iterate through the provided model dictionaries with their types and metrics
    for model_dict, type_of_balancing in model_dicts_list:
        for model_name, model_data in model_dict.items():
            model_identifier = f"{model_name} - {type_of_balancing}"
            # Skip if this model has already been processed
            if model_identifier in processed_models:
                continue 
            processed_models.add(model_identifier)
            # Get the model and make prediction
            model = model_data["model"]

            # Get the threshold for the current model
            model_threshold = thresholds.get(model_identifier, None)
            y_pred = model_data["y_pred"]
            matrix = confusion_matrix(y_test, y_pred)

            # Skip if threshold is None or calculate y_pred_adjusted based on the threshold found
            if model_threshold is None:
                continue
            elif model_threshold != 0.5:
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                y_pred_adjusted = (y_pred_prob >= model_threshold).astype(int)
                matrix = confusion_matrix(y_test, y_pred_adjusted)

            # Plot the first heatmap (absolute values)
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax[plot_index])
            ax[plot_index].set_title(f"{model_identifier} - Absolute Values (Threshold = {model_threshold:.2f})")
            ax[plot_index].set_xlabel("Predicted Values")
            ax[plot_index].set_ylabel("Actual Values")
            plot_index += 1

            # Plot the second heatmap (relative values)
            sns.heatmap(matrix / np.sum(matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[plot_index])
            ax[plot_index].set_title("Relative Values")
            ax[plot_index].set_xlabel("Predicted Values")
            ax[plot_index].set_ylabel("Actual Values")
            plot_index += 1

            # Stop if we have filled all subplots
            if plot_index >= rows * columns:
                break

    fig.tight_layout()
    plt.show()