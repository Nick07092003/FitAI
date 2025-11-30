import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)
app.secret_key = "fit-ai-secret"

# Stores last fitness output so form + results stay visible
last_fitness = None


# =====================================================
#   LOAD DATASET FOR FITNESS PLAN
# =====================================================
df_mega = None
try:
    df_mega = pd.read_csv("megaGymDataset.csv")

    if "Unnamed: 0" in df_mega.columns:
        df_mega.drop("Unnamed: 0", axis=1, inplace=True)

    df_mega.dropna(subset=["Rating"], inplace=True)

    for col in df_mega.select_dtypes(include=["object"]).columns:
        df_mega[col] = df_mega[col].str.replace("_", " ")

    print("MegaGym dataset loaded.")
except Exception as e:
    print("❌ ERROR loading megaGymDataset.csv:", e)


# =====================================================
#   LOAD ML MODELS
# =====================================================

label_encoders = None
title_model = None
equipment_model = None
level_model = None
body_parts = []

try:
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    title_model = pickle.load(open("models/title_model.pkl", "rb"))
    equipment_model = pickle.load(open("models/equipment_model.pkl", "rb"))
    level_model = pickle.load(open("models/level_model.pkl", "rb"))

    body_parts = list(label_encoders["BodyPart"].classes_)
    print("ML models loaded.")
except Exception as e:
    print("❌ ERROR Loading ML Models:", e)


# =====================================================
#   FITNESS PLAN FUNCTIONS
# =====================================================

BFP_STANDARDS_MALE = {
    'Essential': (2, 5), 'Athletes': (6, 13), 'Fitness': (14, 17),
    'Acceptable': (18, 24), 'Obese': (25, 100)
}
BFP_STANDARDS_FEMALE = {
    'Essential': (10, 13), 'Athletes': (14, 20), 'Fitness': (21, 24),
    'Acceptable': (25, 31), 'Obese': (32, 100)
}

def calculate_bmi_case(weight, height):
    bmi = weight / (height * height)
    if bmi < 16: case = "sever thinness"
    elif bmi < 17: case = "moderate thinness"
    elif bmi < 18.5: case = "mild thinness"
    elif bmi < 25: case = "normal"
    elif bmi < 30: case = "over weight"
    elif bmi < 35: case = "obese"
    else: case = "severe obese"
    return bmi, case

def estimate_bfp(bmi, gender, age):
    g = 1 if gender == "male" else 0
    return max(0, (1.20*bmi + 0.23*age - 10.8*g - 5.4))

def bfp_case(bfp, gender):
    table = BFP_STANDARDS_MALE if gender == "male" else BFP_STANDARDS_FEMALE
    for c,(lo,hi) in table.items():
        if lo <= bfp <= hi:
            return c
    return "Obese"

def get_plan(bmi_case):
    mapping = {
        "sever thinness": 1, "moderate thinness": 2, "mild thinness": 3,
        "normal": 4, "over weight": 5, "obese": 6, "severe obese": 7
    }
    return mapping[bmi_case]

def recommend_exercises(plan):
    if df_mega is None:
        return []

    if plan in [1,2,3]:
        f = df_mega[
            (df_mega.Type.isin(["Strength","Olympic Weightlifting"])) &
            (df_mega.Level.isin(["Beginner","Intermediate"])) &
            (df_mega.BodyPart.isin(["Chest","Back","Legs","Shoulders"]))
        ]
    elif plan == 4:
        f = df_mega[df_mega.Level == "Intermediate"]
    elif plan in [5,6]:
        f = df_mega[
            (df_mega.Type.isin(["Cardio","Strength"])) &
            (df_mega.BodyPart.isin(["Legs","Glutes","Full Body","Abdominals"]))
        ]
    else:
        f = df_mega[df_mega.Level == "Beginner"]

    return f.sort_values("Rating", ascending=False).head(10).to_dict("records")


# =====================================================
#   MAIN HTML TEMPLATE (Vertical layout)
# =====================================================

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FitAI App</title>

    <!-- BOOTSTRAP -->
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">

    <!-- GOOGLE FONT -->
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap"
          rel="stylesheet">

    <style>
        body {
            background: #0e0e0e;
            color: #f5f5f5;
            font-family: "Poppins", sans-serif;
        }

        h1, h3, h4, h5 {
            font-weight: 700;
        }

        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ff0033, #ff1a75);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 8px rgba(255, 0, 50, 0.4);
        }

        /* Glass Red Cards */
        .section-card {
            background: rgba(22, 22, 22, 0.6);
            border: 1px solid rgba(255, 0, 40, 0.25);
            border-radius: 18px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(255, 0, 40, 0.15);
            backdrop-filter: blur(10px);
            transition: 0.3s ease;
        }

        .section-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(255, 0, 40, 0.25);
        }

        .btn-red {
            background: linear-gradient(90deg, #ff0033, #b30026);
            border: none;
            padding: 12px;
            font-weight: 600;
            border-radius: 50px;
            color: #fff;
            transition: 0.25s;
        }

        .btn-red:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(255, 0, 40, 0.5);
        }

        .btn-green {
            background: linear-gradient(90deg, #00c46a, #008542);
            border: none;
            padding: 12px;
            font-weight: 600;
            border-radius: 50px;
            color: white;
            transition: 0.25s;
        }

        .btn-green:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(0, 255, 130, 0.5);
        }

        label {
            margin-top: 10px;
            font-weight: 600;
        }

        /* TABLE THEME */
        .table {
            color: #fff;
        }
        thead {
            background: #ff0033 !important;
        }
        tbody tr {
            background: rgba(255, 255, 255, 0.05);
        }
        tbody tr:hover {
            background: rgba(255, 0, 40, 0.25);
            transition: 0.3s;
        }
    </style>
</head>

<body>

<div class="container py-4">
    
    <h1 class="text-center main-title mb-4">
        🏋️ FitAI
    </h1>
    <h2 class="text-center main-title mb-4">
        Personalized Workout Recommender
    </h2>

    <!-- ================== FITNESS PLAN SECTION ================== -->
    <div class="section-card mb-4">
        <h3 class="text-danger fw-bold">1️⃣ BMI • BFP — Fitness Plan Generator</h3>

        <form action="/fitness" method="POST">

            <label>Weight (kg)</label>
            <input name="weight" class="form-control bg-dark text-light border-danger"
                   type="number" step="0.1" value="{{ form_weight }}" required>

            <label>Height (m)</label>
            <input name="height" class="form-control bg-dark text-light border-danger"
                   type="number" step="0.01" value="{{ form_height }}" required>

            <label>Age</label>
            <input name="age" class="form-control bg-dark text-light border-danger"
                   type="number" value="{{ form_age }}" required>

            <label>Gender</label>
            <select name="gender" class="form-control bg-dark text-light border-danger">
                <option value="male" {% if form_gender=='male' %}selected{% endif %}>Male</option>
                <option value="female" {% if form_gender=='female' %}selected{% endif %}>Female</option>
            </select>

            <button class="btn-red w-100 mt-4">Generate Fitness Plan</button>
        </form>

        {% if fitness %}
            <hr>
            <h4 class="text-warning">Your Fitness Report</h4>
            <p><b>BMI:</b> {{ fitness.bmi }} ({{ fitness.bmi_case }})</p>
            <p><b>BFP:</b> {{ fitness.bfp }}% ({{ fitness.bfp_case }})</p>
            <p><b>Assigned Plan:</b> Plan {{ fitness.plan }}</p>

            <h5 class="mt-4 text-info">🔥 Recommended Exercises</h5>

            <table class="table table-bordered mt-3">
                <thead>
                    <tr>
                        <th>Exercise</th>
                        <th>Body Part</th>
                        <th>Equipment</th>
                        <th>Level</th>
                    </tr>
                </thead>
                <tbody>
                    {% for ex in fitness.exercises %}
                    <tr>
                        <td>{{ ex["Title"] }}</td>
                        <td>{{ ex["BodyPart"] }}</td>
                        <td>{{ ex["Equipment"] }}</td>
                        <td>{{ ex["Level"] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endif %}
    </div>


    <!-- ================== ML PREDICTION SECTION ================== -->
    <div class="section-card">
        <h3 class="text-success fw-bold">2️⃣ AI-Based Exercise Prediction</h3>

        <form action="/ml_predict" method="POST">
            <label>Select Body Part</label>
            <select name="bodypart" class="form-control bg-dark text-light border-success">
                {% for bp in body_parts %}
                <option value="{{ bp }}">{{ bp }}</option>
                {% endfor %}
            </select>

            <button class="btn-green w-100 mt-4">Predict Exercise</button>
        </form>

        {% if ml %}
            <hr>
            <h5 class="text-primary">AI Prediction Result</h5>
            <p><b>Exercise:</b> {{ ml.title }}</p>
            <p><b>Equipment:</b> {{ ml.equipment }}</p>
            <p><b>Level:</b> {{ ml.level }}</p>
        {% endif %}
    </div>

</div>

</body>
</html>
"""



# =====================================================
#   ROUTES
# =====================================================

@app.route("/")
def home():
    return render_template_string(
        HOME_HTML,
        fitness=last_fitness,
        ml=None,
        body_parts=body_parts,
        form_weight="",
        form_height="",
        form_age="",
        form_gender="male"
    )


@app.route("/fitness", methods=["POST"])
def fitness_route():
    global last_fitness

    weight = float(request.form["weight"])
    height = float(request.form["height"])
    age = int(request.form["age"])
    gender = request.form["gender"]

    bmi, bmi_case = calculate_bmi_case(weight, height)
    bfp = estimate_bfp(bmi, gender, age)
    bfp_c = bfp_case(bfp, gender)
    plan = get_plan(bmi_case)

    last_fitness = {
        "bmi": round(bmi, 2),
        "bmi_case": bmi_case,
        "bfp": round(bfp, 2),
        "bfp_case": bfp_c,
        "plan": plan,
        "exercises": recommend_exercises(plan)
    }

    return render_template_string(
        HOME_HTML,
        fitness=last_fitness,
        ml=None,
        body_parts=body_parts,
        form_weight=weight,
        form_height=height,
        form_age=age,
        form_gender=gender
    )


@app.route("/ml_predict", methods=["POST"])
def ml_predict_route():
    bodypart = request.form["bodypart"]

    encoded = label_encoders["BodyPart"].transform([bodypart])[0]

    title = title_model.predict([[encoded]])[0]
    equipment = equipment_model.predict([[encoded]])[0]
    level = level_model.predict([[encoded]])[0]

    ml_data = {
        "title": label_encoders["Title"].inverse_transform([title])[0],
        "equipment": label_encoders["Equipment"].inverse_transform([equipment])[0],
        "level": label_encoders["Level"].inverse_transform([level])[0]
    }

    # KEEP form values if user already submitted fitness plan
    if last_fitness:
        fw = request.args.get("weight", "")
        fh = request.args.get("height", "")
        fa = request.args.get("age", "")
        fg = request.args.get("gender", "male")
    else:
        fw = fh = fa = ""
        fg = "male"

    return render_template_string(
        HOME_HTML,
        fitness=last_fitness,   # <-- keep fitness result
        ml=ml_data,
        body_parts=body_parts,
        form_weight=fw,
        form_height=fh,
        form_age=fa,
        form_gender=fg
    )


if __name__ == "__main__":
    app.run(debug=True)
