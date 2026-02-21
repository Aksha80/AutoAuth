import os
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model/classifier.pkl"

class AuthML:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_on_synthetic(self, df):
        X = df[['duration_months', 'severity_score', 'previous_therapy', 'policy_match_score']]
        y = df['approved']
        self.model.fit(X, y)

        os.makedirs("model", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

    def predict(self, features):
        prob = self.model.predict_proba([features])[0][1]
        importance = self.model.feature_importances_
        return prob, importance