import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic data for the model."""
    np.random.seed(42)
    
    data = {
        'CGPA': np.random.uniform(5.0, 10.0, num_samples),
        'Programming_Skill': np.random.randint(1, 11, num_samples),
        'Technical_Skill': np.random.randint(1, 11, num_samples),
        'Communication_Skill': np.random.randint(1, 11, num_samples),
        'Aptitude_Score': np.random.randint(40, 101, num_samples),
        'Study_Hours': np.random.uniform(5, 60, num_samples),
        'Projects': np.random.randint(0, 8, num_samples),
        'Internships': np.random.randint(0, 4, num_samples),
        'Backlogs': np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
        'Attendance_Percentage': np.random.uniform(50, 100, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 2. Create Target Variable "Skill_Score"
    df['Raw_Skill_Score'] = (
        (df['CGPA'] * 10 * 0.2) +
        (df['Programming_Skill'] * 10 * 0.15) +
        (df['Technical_Skill'] * 10 * 0.15) +
        (df['Communication_Skill'] * 10 * 0.1) +
        (df['Aptitude_Score'] * 0.1) +
        (df['Projects'].clip(upper=5) * 20 * 0.1) +
        (df['Internships'].clip(upper=2) * 50 * 0.1) +
        ((df['Attendance_Percentage'] / 100) * 100 * 0.05) -
        (df['Backlogs'] * 5)
    )
    
    # Normalize final score to range 0-100 based on theoretical min/max
    # Max possible raw score = 95, min possible (assuming no negative) can be floored at 0
    df['Skill_Score'] = (df['Raw_Skill_Score'] / 95.0) * 100
    df['Skill_Score'] = df['Skill_Score'].clip(lower=0, upper=100)
    
    # 3. Create Placement_Status
    df['Placement_Status'] = (df['Skill_Score'] >= 80).astype(int)
    
    # Drop raw score as it's no longer needed
    df.drop(columns=['Raw_Skill_Score'], inplace=True)
    
    return df

def train_and_evaluate():
    # 1. Generate Dataset
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(5000)
    
    # 4. Preprocessing
    features = [
        'CGPA', 'Programming_Skill', 'Technical_Skill', 'Communication_Skill',
        'Aptitude_Score', 'Study_Hours', 'Projects', 'Internships',
        'Backlogs', 'Attendance_Percentage'
    ]
    
    X = df[features]
    y_score = df['Skill_Score']
    
    # Handle missing values (though synthetic has none, doing it for safety)
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_score, test_size=0.2, random_state=42)
    
    # 5. Train Models
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 6. Evaluate
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Linear Regression R2 Score: {r2_score(y_test, lr_preds):.4f}")
    print(f"Linear Regression MAE: {mean_absolute_error(y_test, lr_preds):.4f}")
    
    print(f"Random Forest R2 Score: {r2_score(y_test, rf_preds):.4f}")
    print(f"Random Forest MAE: {mean_absolute_error(y_test, rf_preds):.4f}")
    
    # Use the better model (Random Forest)
    best_model = rf_model
    
    # 8. Save model
    print("\nSaving scaler and model...")
    with open('skill_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('skill_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    # 9. Feature Importance
    print("\n--- Feature Importance (Random Forest) ---")
    importances = best_model.feature_importances_
    for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"{name}: {imp:.4f}")
        
    # 10. Test with Sample
    # (CGPA=9.5, Skills=9, Projects=5, Internships=2, Backlogs=0)
    # Filling in missing features with averages
    print("\n--- Testing Sample ---")
    sample_data = pd.DataFrame([{
        'CGPA': 9.5,
        'Programming_Skill': 9,
        'Technical_Skill': 9,
        'Communication_Skill': 9,
        'Aptitude_Score': 90,
        'Study_Hours': 30,
        'Projects': 5,
        'Internships': 2,
        'Backlogs': 0,
        'Attendance_Percentage': 95
    }])
    
    sample_scaled = scaler.transform(sample_data)
    sample_pred = best_model.predict(sample_scaled)[0]
    sample_status = "Placed" if sample_pred >= 80 else "Not Placed"
    
    print(f"Sample Input:\n{sample_data.iloc[0].to_dict()}")
    print(f"Predicted Skill Score: {sample_pred:.2f}")
    print(f"Placement Status: {sample_status}")

if __name__ == "__main__":
    train_and_evaluate()
