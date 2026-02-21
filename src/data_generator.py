import pandas as pd
import numpy as np

def generate_synthetic_data(n=500):
    data = {
        'duration_months': np.random.randint(0, 12, n),
        'severity_score': np.random.randint(1, 4, n),
        'previous_therapy': np.random.randint(0, 2, n),
        'policy_match_score': np.random.uniform(0.1, 1.0, n)
    }
    df = pd.DataFrame(data)
    # Simple logic for approval target:
    # Approved if (duration > 3 AND severity > 1) OR policy_match > 0.8
    df['approved'] = ((df['duration_months'] > 3) & (df['severity_score'] > 1) | (df['policy_match_score'] > 0.8)).astype(int)
    df.to_csv('data/historical_data.csv', index=False)
    return df