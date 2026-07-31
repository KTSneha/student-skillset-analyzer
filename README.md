# Student Skillset Analyzer

A comprehensive Python-Flask data science product for student skill scoring, placement eligibility prediction, smart recommendations, and analytics.

## Features

- **User Accounts**: Student registration, login, and robust profile management.
- **Machine Learning**: Dynamic skill score prediction and automated placement eligibility algorithms.
- **Admin Dashboard**: Administrator privileges for dataset uploads, retraining the ML models on-the-fly, and system analytics.
- **Smart Engine**: Rule-based recommendation engine for highlighting areas to improve.
- **Reports**: Downloadable PDF report generation for students.
- **Beautiful UI**: Modern, glassmorphic UI with dynamic Plotly charts and visual dashboards.
- **Data Persistence**: Complete local state management with SQLite, pickle model persistence, and CSV integration.

## Tech Stack

- **Backend**: Flask, Flask-Login, Flask-WTF, Flask-SQLAlchemy
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Frontend**: Bootstrap 5, Custom CSS, Plotly.js
- **Database**: SQLite
- **Utilities**: xhtml2pdf (for PDF generation)

---

## 🚀 Setup & Installation Guide

Follow these exact commands to successfully run the project on any local machine.

### 1. Clone the Repository
If you haven't already, clone or download the project folder to your system, and navigate into it:
```bash
# Navigate to the project directory
cd Student-Skillset-Analyzer
```
*(Note: Replace `Student-Skillset-Analyzer` with the actual folder name if it differs on your machine).*

### 2. Create a Virtual Environment (Highly Recommended)
Creating a virtual environment ensures that the project's dependencies don't interfere with your system's global Python packages.

**For Windows:**
```bash
python -m venv venv
```

**For Mac/Linux:**
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
You must activate the virtual environment before installing the requirements.

**For Windows (Command Prompt / PowerShell):**
```bash
.\venv\Scripts\activate
```

**For Mac/Linux:**
```bash
source venv/bin/activate
```
*(When activated, your terminal prompt will usually be prefixed with `(venv)`).*

### 4. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 5. Run the Application
Start the Flask local development server:
```bash
python app.py
```
*(Mac/Linux users may need to use `python3 app.py` depending on their alias setup).*

### 6. Access the Web App
Open your web browser and go to:
**http://127.0.0.1:5000**

---


---

## Project Structure

- `app.py`: Main Flask application, routing, and controller logic
- `models.py`: Database schema models for Users and Student Profiles
- `ml_utils.py`: ML pipeline (preprocessing, training, predicting, and model persistence)
- `recommender.py`: Rule-based recommendation engine module
- `forms.py`: Flask-WTF forms for validation and rendering
- `templates/`: HTML views and Jinja2 templates
- `static/`: Custom CSS styling and static assets
- `Data/`: Directory storing trained `.pkl` model artifacts and synthetic dataset files
