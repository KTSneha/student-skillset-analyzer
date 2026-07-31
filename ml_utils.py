import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from config import Config

FEATURE_COLUMNS = [
    "CGPA",
    "Programming_Skill",
    "Technical_Skill",
    "Communication_Skill",
    "Aptitude_Score",
    "Study_Hours_Per_Week",
    "Projects",
    "Internships",
    "Backlogs",
    "Attendance_Percent",
]

TARGET_SCORE = "Skill_Score"
TARGET_STATUS = "Placement_Status"

def ensure_directories():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)

def generate_synthetic_data(num_samples=2000):
    np.random.seed(42)
    data = {
        'CGPA': np.random.uniform(5.0, 10.0, num_samples),
        'Programming_Skill': np.random.randint(1, 11, num_samples),
        'Technical_Skill': np.random.randint(1, 11, num_samples),
        'Communication_Skill': np.random.randint(1, 11, num_samples),
        'Aptitude_Score': np.random.randint(40, 101, num_samples),
        'Study_Hours_Per_Week': np.random.uniform(5, 60, num_samples),
        'Projects': np.random.randint(0, 8, num_samples),
        'Internships': np.random.randint(0, 4, num_samples),
        'Backlogs': np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
        'Attendance_Percent': np.random.uniform(50, 100, num_samples)
    }
    df = pd.DataFrame(data)
    return df

def load_dataset(path=None):
    path = path or Config.CSV_DATA_PATH
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return generate_synthetic_data(100)

def preprocess_data(df):
    df = df.copy()
    
    # Ensure all required features are present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            if col == "Technical_Skill" and "Programming_Skill" in df.columns:
                df[col] = df["Programming_Skill"] # fallback
            elif col == "Study_Hours_Per_Week" and "Study_Hours" in df.columns:
                df[col] = df["Study_Hours"]
            elif col == "Attendance_Percent" and "Attendance_Percentage" in df.columns:
                df[col] = df["Attendance_Percentage"]
            else:
                df[col] = 0
                
    df = df.fillna(df.median())
    
    # Calculate Target Variable "Skill_Score"
    df['Raw_Skill_Score'] = (
        (df['CGPA'] * 10 * 0.2) +
        (df['Programming_Skill'] * 10 * 0.15) +
        (df['Technical_Skill'] * 10 * 0.15) +
        (df['Communication_Skill'] * 10 * 0.1) +
        (df['Aptitude_Score'] * 0.1) +
        (df['Projects'].clip(upper=5) * 20 * 0.1) +
        (df['Internships'].clip(upper=2) * 50 * 0.1) +
        ((df['Attendance_Percent'] / 100) * 100 * 0.05) -
        (df['Backlogs'] * 5)
    )
    
    # Normalize final score to range 0-100 based on max possible score (95)
    df[TARGET_SCORE] = (df['Raw_Skill_Score'] / 95.0) * 100
    df[TARGET_SCORE] = df[TARGET_SCORE].clip(lower=0, upper=100)
    
    # Create Placement_Status
    df[TARGET_STATUS] = (df[TARGET_SCORE] >= 75).astype(int)
    
    df.drop(columns=['Raw_Skill_Score'], inplace=True, errors='ignore')
    
    # Save processed dataset
    ensure_directories()
    df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    return df

def train_models(df=None):
    if df is None or len(df) < 100:
        df = generate_synthetic_data(2000)
        
    df = preprocess_data(df)
    X = df[FEATURE_COLUMNS]
    y_score = df[TARGET_SCORE]
    y_status = df[TARGET_STATUS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_score_train, y_score_test, y_status_train, y_status_test = train_test_split(
        X_scaled, y_score, y_status, test_size=0.2, random_state=42
    )

    # Linear Regression (for Skill Score)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_score_train)
    
    # Random Forest Regressor (better accuracy)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_score_train)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_status_train)

    score_predictions = rf_model.predict(X_test)
    status_predictions = rf_classifier.predict(X_test)

    metrics = {
        "score_r2": r2_score(y_score_test, score_predictions),
        "score_mae": mean_absolute_error(y_score_test, score_predictions),
        "status_accuracy": accuracy_score(y_status_test, status_predictions)
    }

    # Save artifacts (use rf_model for score so feature_importances_ is available)
    save_artifacts(scaler, rf_model, rf_classifier)
    return metrics

def save_artifacts(scaler, score_model, status_model):
    ensure_directories()
    with open(os.path.join(Config.MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(Config.MODEL_DIR, "score_model.pkl"), "wb") as f:
        pickle.dump(score_model, f)
    with open(os.path.join(Config.MODEL_DIR, "status_model.pkl"), "wb") as f:
        pickle.dump(status_model, f)

def load_artifacts():
    ensure_directories()
    scaler_path = os.path.join(Config.MODEL_DIR, "scaler.pkl")
    score_path = os.path.join(Config.MODEL_DIR, "score_model.pkl")
    status_path = os.path.join(Config.MODEL_DIR, "status_model.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(score_path) or not os.path.exists(status_path):
        return None, None, None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(score_path, "rb") as f:
        score_model = pickle.load(f)
    with open(status_path, "rb") as f:
        status_model = pickle.load(f)
    return scaler, score_model, status_model

def predict_from_features(features):
    scaler, score_model, status_model = load_artifacts()
    if scaler is None or score_model is None or status_model is None:
        train_models()
        scaler, score_model, status_model = load_artifacts()
    
    mapped_features = {
        "CGPA": features.get("CGPA", 7.0),
        "Programming_Skill": features.get("Programming_Skill", 5),
        "Technical_Skill": features.get("Technical_Skill", features.get("Programming_Skill", 5)),
        "Communication_Skill": features.get("Communication_Skill", 5),
        "Aptitude_Score": features.get("Aptitude_Score", 50),
        "Study_Hours_Per_Week": features.get("Study_Hours_Per_Week", 10),
        "Projects": features.get("Projects", 0),
        "Internships": features.get("Internships", 0),
        "Backlogs": features.get("Backlogs", 0),
        "Attendance_Percent": features.get("Attendance_Percent", 80),
    }
    
    data = pd.DataFrame([mapped_features], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(data)
    score_prediction = score_model.predict(X_scaled)[0]
    status_prediction = bool(score_prediction >= 75)
    
    return {
        "skill_score": float(np.clip(score_prediction, 0, 100)),
        "placement_status": status_prediction,
        "status_label": "Eligible" if status_prediction else "Not Eligible",
    }

def feature_importance():
    _, score_model, _ = load_artifacts()
    if score_model is None:
        return []
    importance = list(zip(FEATURE_COLUMNS, score_model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    return [
        {"feature": feature, "importance": float(score)}
        for feature, score in importance
    ]
