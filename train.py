import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Generate sample student data
np.random.seed(42)
n = 500

study_hours = np.random.uniform(0, 10, n)
attendance = np.random.uniform(30, 100, n)
internal_marks = np.random.uniform(0, 100, n)

# Sample rule for generating Pass/Fail labels
score = (
    study_hours * 4
    + attendance * 0.2
    + internal_marks * 0.5
)

results = (score >= 55).astype(int)

X = np.column_stack(
    (study_hours, attendance, internal_marks)
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, results)

joblib.dump(model, "student_model.pkl")

print("Model trained successfully.")
print("Model saved as student_model.pkl")
