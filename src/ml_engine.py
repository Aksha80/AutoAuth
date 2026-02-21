from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib
import pandas as pd

MODEL_PATH = "model/classifier.pkl"

class AuthML:

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = LogisticRegression()

    def train_on_synthetic(self, df):

        X = df[['duration_months','severity_score',
                'previous_therapy','policy_match_score']]
        y = df['approved']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Model Accuracy:", round(acc*100,2), "%")

        os.makedirs("model", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

    def predict(self, features):
        # Convert to DataFrame with correct feature names
        df = pd.DataFrame([features], columns=self.model.feature_names_in_)

        prob = self.model.predict_proba(df)[0][1]

        # Soft clipping
        prob = 0.05 + (prob * 0.90)
        return prob