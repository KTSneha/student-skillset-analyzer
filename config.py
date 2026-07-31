import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(DATA_DIR, "trained_models")
CSV_DATA_PATH = os.path.join(DATA_DIR, "student_skillset_ready_dataset.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_dataset.csv")

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'skta.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATA_DIR = DATA_DIR
    MODEL_DIR = MODEL_DIR
    CSV_DATA_PATH = CSV_DATA_PATH
    PROCESSED_DATA_PATH = PROCESSED_DATA_PATH
    ALLOWED_EXTENSIONS = {"csv"}
