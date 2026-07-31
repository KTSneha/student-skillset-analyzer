from datetime import datetime
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="student")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile = db.relationship("StudentProfile", backref="user", uselist=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class StudentProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    full_name = db.Column(db.String(120), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    cgpa = db.Column(db.Float, nullable=True)
    programming_skill = db.Column(db.Integer, nullable=True)
    communication_skill = db.Column(db.Integer, nullable=True)
    aptitude_score = db.Column(db.Float, nullable=True)
    internships = db.Column(db.Integer, nullable=True)
    projects = db.Column(db.Integer, nullable=True)
    backlogs = db.Column(db.Integer, nullable=True)
    attendance_percent = db.Column(db.Float, nullable=True)
    certifications = db.Column(db.String(500), nullable=True)
    study_hours_per_week = db.Column(db.Float, nullable=True)
    placed = db.Column(db.Boolean, default=False)
    salary_lpa = db.Column(db.Float, nullable=True)
    resume_text = db.Column(db.Text, nullable=True)
    skill_score = db.Column(db.Float, nullable=True)
    placement_status = db.Column(db.Boolean, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    score_history = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def skill_summary(self):
        return {
            "Programming": self.programming_skill or 0,
            "Communication": self.communication_skill or 0,
            "Aptitude": self.aptitude_score or 0,
            "Projects": self.projects or 0,
            "Internships": self.internships or 0,
            "Study Hours/Week": self.study_hours_per_week or 0,
        }

    def add_score_history(self, score):
        try:
            history = json.loads(self.score_history or "[]")
        except ValueError:
            history = []
        history.append({"score": float(score), "timestamp": datetime.utcnow().isoformat()})
        self.score_history = json.dumps(history)

    def get_score_history(self):
        try:
            return json.loads(self.score_history or "[]")
        except ValueError:
            return []
