from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, IntegerField, FloatField, TextAreaField, FileField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, NumberRange, Optional

class RegistrationForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo("password")])
    submit = SubmitField("Register")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember me")
    submit = SubmitField("Login")

class ProfileForm(FlaskForm):
    full_name = StringField("Full Name", validators=[DataRequired()])
    gender = SelectField("Gender", choices=[('', 'Select Gender'), ('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], validators=[Optional()])
    city = StringField("City", validators=[Optional()])
    cgpa = FloatField("CGPA", validators=[Optional(), NumberRange(min=0, max=10)])
    programming_skill = IntegerField("Programming Skill (1-10)", validators=[Optional(), NumberRange(min=1, max=10)])
    communication_skill = IntegerField("Communication Skill (1-10)", validators=[Optional(), NumberRange(min=1, max=10)])
    aptitude_score = FloatField("Aptitude Score", validators=[Optional(), NumberRange(min=0, max=100)])
    internships = IntegerField("Number of Internships", validators=[Optional(), NumberRange(min=0, max=10)])
    projects = IntegerField("Number of Projects", validators=[Optional(), NumberRange(min=0, max=50)])
    backlogs = IntegerField("Number of Backlogs", validators=[Optional(), NumberRange(min=0, max=20)])
    attendance_percent = FloatField("Attendance Percentage", validators=[Optional(), NumberRange(min=0, max=100)])
    certifications = StringField("Certifications (comma-separated)", validators=[Optional()])
    study_hours_per_week = FloatField("Study Hours Per Week", validators=[Optional(), NumberRange(min=0, max=168)])
    placed = BooleanField("Currently Placed")
    salary_lpa = FloatField("Salary (LPA)", validators=[Optional(), NumberRange(min=0, max=1000)])
    resume_text = TextAreaField("Resume / Experience Summary", validators=[Optional()])
    submit = SubmitField("Save Profile")

class CSVUploadForm(FlaskForm):
    dataset = FileField("Dataset CSV", validators=[DataRequired()])
    submit = SubmitField("Upload Dataset")
