import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

            logging.info("Splitting training and test input data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(objective="reg:squarederror"),
                "CatBoost": CatBoostRegressor(
                    verbose=False,
                allow_writing_files=False
                       ),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {

                "Linear Regression": {},

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20]
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },
            }

            model_report = evaluate_model(
                     X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                     y_test=y_test,
                     models=models,
                     param=params,
                    )

            best_model_name = max(
                  model_report,
                   key=lambda x: model_report[x]["score"]
                     )

            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]
            
            print(f"Best Model: {best_model_name}")
            print(f"Best Score: {best_model_score}")
            

            
            if best_model_score < 0.6:
                raise Exception("No best model found")

            logging.info(f"Best model found: {best_model_name}")
            logging.info(f"Best R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)