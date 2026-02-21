from src.data_generator import generate_synthetic_data
from src.ml_engine import AuthML

print("Generating data...")
df = generate_synthetic_data(1000)

print("Training model...")
ml = AuthML()
ml.train_on_synthetic(df)

print("Model trained successfully!")