import os
import json
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, flash, request, send_file, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from models import db, User, StudentProfile
from forms import RegistrationForm, LoginForm, ProfileForm, CSVUploadForm
from config import Config
from ml_utils import load_dataset, preprocess_data, train_models, load_artifacts, predict_from_features, feature_importance
from recommender import build_recommendations
from xhtml2pdf import pisa


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.CSV_DATA_PATH), exist_ok=True)
        db.create_all()
        if not User.query.filter_by(email="admin@gmail.com").first():
            admin = User(email="admin@gmail.com", role="admin")
            admin.set_password("Admin123")
            db.session.add(admin)
            db.session.commit()
        # Train models if they don't exist
        if not os.path.exists(os.path.join(Config.MODEL_DIR, "scaler.pkl")):
            try:
                df = load_dataset(Config.CSV_DATA_PATH)
                train_models(df)
                print("Models trained successfully on startup")
            except Exception as e:
                print(f"Failed to train models on startup: {e}")
        if not os.path.exists(Config.CSV_DATA_PATH):
            try:
                sample = load_dataset(Config.CSV_DATA_PATH)
                sample.to_csv(Config.CSV_DATA_PATH, index=False)
            except FileNotFoundError:
                pass

    @app.route("/")
    def home():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        form = RegistrationForm()
        if form.validate_on_submit():
            if User.query.filter_by(email=form.email.data).first():
                flash("Email already exists.", "warning")
                return redirect(url_for("register"))
            user = User(email=form.email.data, role="student")
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            profile = StudentProfile(user_id=user.id)
            db.session.add(profile)
            db.session.commit()
            flash("Registration successful. Log in to continue.", "success")
            return redirect(url_for("login"))
        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.check_password(form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for("dashboard"))
            flash("Invalid credentials.", "danger")
        return render_template("login.html", form=form)

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        if current_user.role == "admin":
            return redirect(url_for("admin_dashboard"))
        profile = current_user.profile
        if not profile:
            flash("Create your profile first.", "warning")
            return redirect(url_for("profile"))

        student_chart = profile.skill_summary()
        score_history = profile.get_score_history()
        recommendations = json.loads(profile.recommendations or "[]")
        weak_areas = {}
        for skill, value in student_chart.items():
            if skill == "Projects" and value < 3:
                weak_areas[skill] = value
            elif skill == "Internships" and value < 1:
                weak_areas[skill] = value
            elif skill == "Aptitude" and value < 60:
                weak_areas[skill] = value
            elif skill == "Study Hours/Week" and value < 10:
                weak_areas[skill] = value
            elif skill in ["Programming", "Communication"] and value < 6:
                weak_areas[skill] = value
        weak_areas_count = len(weak_areas)
        return render_template(
            "dashboard.html",
            profile=profile,
            skill_chart=student_chart,
            score_history=score_history,
            recommendations=recommendations,
            weak_areas=weak_areas,
            weak_areas_count=weak_areas_count,
        )

    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        if current_user.role == "admin":
            return redirect(url_for("admin_dashboard"))
        profile = current_user.profile
        if profile is None:
            profile = StudentProfile(user_id=current_user.id)
            db.session.add(profile)
            db.session.commit()
        form = ProfileForm(obj=profile)
        if form.validate_on_submit():
            form.populate_obj(profile)
            profile.user_id = current_user.id
            profile.recommendations = json.dumps(build_recommendations(profile))
            features = {
                "CGPA": profile.cgpa or 0,
                "Programming_Skill": profile.programming_skill or 0,
                "Communication_Skill": profile.communication_skill or 0,
                "Aptitude_Score": profile.aptitude_score or 0,
                "Internships": profile.internships or 0,
                "Projects": profile.projects or 0,
                "Backlogs": profile.backlogs or 0,
                "Attendance_Percent": profile.attendance_percent or 0,
                "Certifications": len(profile.certifications.split(',')) if profile.certifications else 0,
                "Study_Hours_Per_Week": profile.study_hours_per_week or 0,
            }
            result = predict_from_features(features)
            profile.skill_score = round(result["skill_score"])
            profile.placement_status = result["placement_status"]
            profile.add_score_history(profile.skill_score)
            db.session.commit()
            flash("Profile saved and score updated.", "success")
            return redirect(url_for("dashboard"))
        elif request.method == "POST":
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"Error in {getattr(form, field).label.text}: {error}", "danger")
        return render_template("profile.html", form=form, profile=profile)

    @app.route("/recommendations")
    @login_required
    def recommendations():
        profile = current_user.profile
        recs = json.loads(profile.recommendations or "[]")
        return render_template("recommendations.html", recommendations=recs)

    @app.route("/admin")
    @login_required
    def admin_dashboard():
        if current_user.role != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        students = StudentProfile.query.all()
        metrics = {
            "average_score": round(sum((s.skill_score or 0) for s in students) / max(1, len(students)), 2),
            "eligible_percent": round(sum(1 for s in students if s.placement_status) / max(1, len(students)) * 100, 2) if students else 0,
            "count": len(students),
        }
        feature_importances = feature_importance()
        return render_template("admin_dashboard.html", students=students, metrics=metrics, feature_importances=feature_importances)

    @app.route("/admin/upload", methods=["GET", "POST"])
    @login_required
    def upload_dataset():
        if current_user.role != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        form = CSVUploadForm()
        if form.validate_on_submit():
            file = form.dataset.data
            filename = secure_filename(file.filename)
            if filename.endswith(".csv"):
                path = os.path.join(Config.DATA_DIR, filename)
                file.save(path)
                df = load_dataset(path)
                preprocess_data(df)
                flash("Dataset uploaded successfully. Retrain the model below.", "success")
                return redirect(url_for("admin_dashboard"))
            flash("Please upload a CSV file.", "warning")
        return render_template("dataset_upload.html", form=form)

    @app.route("/admin/retrain")
    @login_required
    def retrain():
        if current_user.role != "admin":
            return jsonify({"error": "Unauthorized"}), 403
        df = load_dataset(Config.CSV_DATA_PATH)
        metrics = train_models(df)
        return render_template("retrain_complete.html", metrics=metrics)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json() or {}
        try:
            result = predict_from_features(data)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/report")
    @login_required
    def report():
        if current_user.role != "student":
            flash("Student report only.", "warning")
            return redirect(url_for("admin_dashboard"))
        profile = current_user.profile
        recommendations = json.loads(profile.recommendations or "[]")
        html = render_template(
            "report.html",
            profile=profile,
            recommendations=recommendations,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
        pdf = BytesIO()
        pisa_status = pisa.CreatePDF(html, dest=pdf)
        pdf.seek(0)
        if pisa_status.err:
            flash("Failed to generate PDF.", "danger")
            return redirect(url_for("dashboard"))
        return send_file(pdf, mimetype="application/pdf", download_name="student_report.pdf")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8000)
