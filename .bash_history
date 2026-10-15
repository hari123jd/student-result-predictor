mkdir -p student-result-predictor/template
mkdir -p cd student-result-predictor
cat > requirements.txt <<'EOF'
flask
scikit-learn
numpy
joblib
EOF

cat > train.py <<'EOF'
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
EOF

cat > app.py <<'EOF'
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "student_model.pkl"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")

# Load the trained model
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            internal_marks = float(request.form["internal_marks"])

            if not 0 <= study_hours <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= attendance <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if not 0 <= internal_marks <= 100:
                raise ValueError("Internal marks must be between 0 and 100.")

            result = model.predict([[
                study_hours,
                attendance,
                internal_marks
            ]])[0]

            prediction = "Pass" if result == 1 else "Fail"

        except (ValueError, TypeError):
            error = "Enter valid numeric values within the allowed ranges."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        version=MODEL_VERSION
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
EOF

cat > templates/index.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Student Result Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            margin: 0;
            padding: 30px;
        }
        .container {
            max-width: 450px;
            margin: 30px auto;
            padding: 25px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 15px #0001;
        }
        h1 { color: #1d4ed8; text-align: center; }
        label { display: block; margin-top: 15px; }
        input {
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            margin-top: 6px;
        }
        button {
            width: 100%;
            padding: 12px;
            margin-top: 20px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 12px;
            background: #dcfce7;
            text-align: center;
            font-weight: bold;
        }
        .error { color: #dc2626; margin-top: 15px; }
        .version { text-align: center; color: #64748b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Student Result Predictor</h1>

        <p>Enter the student's details below.</p>

        <form method="POST">
            <label for="study_hours">Study Hours</label>
            <input id="study_hours" name="study_hours"
                   type="number" min="0" max="24"
                   step="0.1" required>

            <label for="attendance">Attendance (%)</label>
            <input id="attendance" name="attendance"
                   type="number" min="0" max="100"
                   step="0.1" required>

            <label for="internal_marks">Internal Marks (%)</label>
            <input id="internal_marks" name="internal_marks"
                   type="number" min="0" max="100"
                   step="0.1" required>

            <button type="submit">Predict Result</button>
        </form>

        {% if prediction %}
        <div class="result">
            Predicted Result: {{ prediction }}
        </div>
        {% endif %}

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <p class="version">Model Version: {{ version }}</p>
    </div>
</body>
</html>
EOF

sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
