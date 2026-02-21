import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n=3000):

    np.random.seed(42)

    duration = np.random.randint(0, 12, n)
    severity = np.random.randint(1, 4, n)
    therapy = np.random.randint(0, 2, n)
    policy_score = np.random.uniform(0.1, 1.0, n)

    # Create weighted approval probability (soft logic)
    weighted_score = (
        0.35 * (duration / 12) +
        0.30 * (severity / 3) +
        0.20 * therapy +
        0.15 * policy_score
    )

    # Add small noise for overlap
    noise = np.random.normal(0, 0.05, n)
    weighted_score = weighted_score + noise

    approved = (weighted_score > 0.5).astype(int)

    df = pd.DataFrame({
        'duration_months': duration,
        'severity_score': severity,
        'previous_therapy': therapy,
        'policy_match_score': policy_score,
        'approved': approved
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/historical_data.csv", index=False)

    return df