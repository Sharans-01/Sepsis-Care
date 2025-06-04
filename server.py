from datetime import datetime
from functools import wraps
from flask import g
import joblib
from flask import Flask, render_template, redirect, url_for, flash,request,make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import  logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from code_files.config import Config
from code_files.forms import RegisterForm, LoginForm, ChangePasswordForm
import pandas as pd
from tensorflow.keras.models import load_model
import pdfkit
import io
from flask import send_file, render_template
from flask import request
import os

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)

@app.before_first_request
def create_tables():
    db.create_all()

# Load the trained model_1
model = load_model('model/MLP_model.h5')
scaler = joblib.load('model/Scaler.pkl')

# Define the features as per the training data
features = [
    'HR', 'Temp', 'Resp', 'O2Sat', 'SBP', 'MAP', 'DBP',
    'WBC', 'Lactate', 'Creatinine', 'BUN', 'Platelets', 'Glucose',
    'pH', 'HCO3', 'PaCO2', 'FiO2', 'PTT', 'Fibrinogen', 'Age',
    'Gender'
]

# Stage 1 Precaution Message
STAGE_1_MESSAGE = """
🟡 <strong>General Precautions, Medications, and Habits for Stage 1 Sepsis (Initial Sepsis):</strong><br>
<strong>PRECAUTIONS:</strong><br>
1. Early Recognition of Infection Signs
Monitor for symptoms like fever, chills, rapid breathing, increased heart rate, and confusion. Promptly identify any signs of infection or systemic inflammation.<br>
2. Timely Medical Evaluation
Seek immediate medical attention if infection symptoms worsen or if systemic signs appear. Early assessment by healthcare professionals is critical.<br>
3. Prompt Administration of Antibiotics
Start broad-spectrum antibiotics as soon as sepsis is suspected to control the infection before it spreads.<br>
4. Maintain Adequate Hydration
Ensure the patient receives sufficient fluids intravenously to support circulation and prevent low blood pressure.<br>
5. Monitor Vital Signs Frequently
Regularly check temperature, heart rate, respiratory rate, and blood pressure to detect any deterioration early.<br>
6. Prevent Infection Spread
Practice strict hygiene, wound care, and aseptic techniques to avoid worsening or secondary infections.<br>
<hr>
<strong>MEDICATIONS:</strong><br>
- <strong>Broad-Spectrum Antibiotics</strong>: Piperacillin-Tazobactam (Zosyn), Meropenem (Merrem), Vancomycin (Vancocin)<br>
- <strong>Vasopressors</strong> (if needed): Norepinephrine (Levophed), Dopamine<br>
- <strong>Fluids</strong>: Normal Saline or Lactated Ringer's<br>
- <strong>Oxygen Therapy</strong>: Maintain SpO2 > 94%<br>
- <strong>Pain/Sedation</strong>: Acetaminophen (Tylenol), Lorazepam (Ativan)<br>
- <strong>Glucose Control</strong>: Insulin if hyperglycemic<br>
- <strong>DVT Prevention</strong>: Enoxaparin (Lovenox)<br>
<hr>
<strong>SUPPLEMENT SCHEDULE:</strong><br>
-> Ceftriaxone – 1-0-1 (1g IV in the morning and night)<br>
-> Piperacillin-Tazobactam – 1-1-1 (4.5g IV three times daily)<br>
-> Meropenem (alternative to Piperacillin-Tazobactam) – 1-1-1 (1g IV three times daily)<br>
-> Vancomycin – 1-0-1 (1g IV every 12 hours)<br>
-> Paracetamol (Acetaminophen) – 1-0-1 (500–1000mg oral in the morning and night, as needed for fever)<br>
-> Ibuprofen (optional/if needed) – 1-0-1 (400mg oral in the morning and night for pain/fever)<br>

"""

# Stage 2 Precaution Message
STAGE_2_MESSAGE = """
🔴 <strong>Precautions and Medications for Stage 2: Severe Sepsis</strong><br><br>
<strong>PRECAUTIONS:</strong><br>
1. Admit patient to ICU for continuous monitoring
    → Vital signs and organ function must be tracked hourly.<br>
2. Start IV fluids aggressively
    → Large volume fluid resuscitation (e.g., Normal Saline) to maintain blood pressure.<br>
3. Administer culture-based targeted antibiotics
    → Use antibiotics based on lab reports; adjust as needed.<br>
4. Begin organ support if needed
    → Oxygen, dialysis, or vasopressors (like norepinephrine) if organs start failing.<br>
5. Identify and remove source of infection
    → Drain abscesses, remove infected devices, or perform surgery if required.<br>
6. Provide nutritional and metabolic support
    → Start enteral/parenteral nutrition and monitor glucose, electrolytes, and kidney function.<br>
<hr>
<strong>MEDICATIONS:</strong><br>
- <strong>Antibiotics</strong>: Metronidazole, Meropenem, Vancomycin, Ceftriaxone or Cefepime<br>
- <strong>Vasopressors (If blood pressure stays low)</strong>: Norepinephrine (1st choice), Dopamine, Vasopressin<br>
- <strong>IV Fluids (Essential for blood pressure)</strong>: Normal Saline or Lactated Ringer's<br>
- <strong>Oxygen Therapy</strong>: Maintain SpO2 > 94%<br>
- <strong>Pain & Sedation</strong>: Acetaminophen (Tylenol), Lorazepam (Ativan)<br>
- <strong>Blood Glucose Control</strong>: Insulin for hyperglycemia<br>
- <strong>DVT Prevention</strong>: Enoxaparin (Lovenox)<br>
<hr>
<strong>SUPPLEMENT SCHEDULE:</strong><br>
1. Meropenem – 1-1-1
    → 1g IV every 8 hours (broad-spectrum antibiotic)<br>
2. Vancomycin – 1-0-1
    → 1g IV every 12 hours (for resistant Gram-positive bacteria)<br>
3. Norepinephrine (vasopressor) – Continuous IV infusion
    → To maintain blood pressure if fluids are not enough<br>
4. Paracetamol (Acetaminophen) – 1-0-1
    → 500–1000mg orally or IV for fever or discomfort<br>
5. Proton Pump Inhibitor (e.g., Pantoprazole) – 1-0-0
    → 40mg IV once daily to prevent gastric ulcers due to stress/sepsis<br>
6. Insulin (if blood sugar is high) – As needed (sliding scale)
    → Tight glucose control is important in sepsis<br>
"""

STAGE_3_MESSAGE = """
🔴 <strong>Precautions and Medications for Stage 3: Septic Shock</strong><br><br>
<strong>PRECAUTIONS:</strong><br>
1. Admit Immediately to ICU with Full Life Support
    → The patient must be in the Intensive Care Unit with ventilator and monitoring support.<br>
2. Start High-Dose IV Vasopressors Quickly
    → Norepinephrine or Epinephrine is used to maintain blood pressure when fluids fail.<br>
3. Continue High-Volume IV Fluid Resuscitation
    → Give fluids like Normal Saline or Lactated Ringer’s to restore circulation.<br>
4. Use Advanced Oxygen Therapy or Mechanical Ventilation
    → Oxygen mask, BiPAP, or intubation may be required for respiratory failure.<br>
5. Administer Strong Combined Antibiotics Immediately
    → Use combinations (e.g., Meropenem + Vancomycin) until culture results guide targeted therapy.<br>
6. Monitor and Support Failing Organs Constantly
    → Kidney (dialysis), liver, lungs (ventilator), heart (vasopressors), and glucose control must all be managed closely.<br>
<hr>
<strong>MEDICATIONS:</strong><br>
- <strong>Broad-Spectrum Combination Antibiotics</strong>: Piperacillin-Tazobactam, Meropenem, Vancomycin, Linezolid (if resistant Gram-positive suspected)<br>
- <strong>Vasopressors (to maintain blood pressure)</strong>: Norepinephrine, Epinephrine, Vasopressin<br>
- <strong>Intravenous Fluids</strong>: Normal Saline, Lactated Ringer’s<br>
- <strong>Antipyretics (to reduce fever and discomfort)</strong>: Paracetamol (Acetaminophen), Avoid NSAIDs if kidney function is poor<br>
- <strong>Oxygen/Ventilation</strong>: High-flow O₂ or Mechanical ventilation if respiratory failure occurs<br>
- <strong>Pain/Sedation</strong>: Acetaminophen, Lorazepam<br>
- <strong>Insulin</strong>: Blood glucose control<br>
- <strong>Anticoagulants</strong>: Enoxaparin (DVT prevention)<br>
<hr>
<strong>SUPPLEMENT SCHEDULE:</strong><br>
-> Meropenem – 1g IV 1-1-1<br>
-> Vancomycin – 1g IV 1-0-1<br>
-> Metronidazole – 500mg IV 1-1-1<br>
-> Linezolid – 600mg IV 1-1-0<br>
-> Hydrocortisone – 50mg IV 1-1-1-1<br>
-> Paracetamol – 500mg 1-0-1<br>
-> Tramadol – 50mg 1-1-1<br>
-> Pantoprazole – 40mg 1-0-0<br>
-> Enoxaparin – 40mg (injection) 0-0-1<br>
-> Insulin – As per blood sugar levels<br>
-> IV Fluids – Normal Saline / RL, continuous<br>
-> Vasopressors (Norepinephrine, etc.) – IV drip (continuous)<br>
"""


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = request.cookies.get('user_id')
        if not user_id:
            return redirect(url_for('login'))

        doctor = Doctor.query.get(int(user_id))
        if not doctor:
            return redirect(url_for('login'))

        g.current_doctor = doctor  # store the doctor for use in views
        return f(*args, **kwargs)
    return decorated_function


class Doctor(db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200),nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    department = db.Column(db.String(200), nullable=False)
    phone_number = db.Column(db.String(15), unique=True, nullable=False)
    password_hash = db.Column(db.String(225), nullable=False)

    # Method to hash the password
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Method to check the password
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Optional: If you want to implement is_active explicitly
    @property
    def is_active(self):
        return True  # Or some condition to check if the doctor is active

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

# Define the Patient model_1
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    ward = db.Column(db.String(20), nullable=False)
    attendee = db.Column(db.String(15), nullable=False)
    address = db.Column(db.Text, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    consultant = db.Column(db.String(100), nullable=False)
    admission_date = db.Column(db.Date, nullable=False)
    blood_sample_report = db.Column(db.Text, nullable=False)
    timeseries_samples = db.Column(db.Text, nullable=False)
    additional_observations = db.Column(db.Text)


class SepsisPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100))
    patient_name = db.Column(db.String(100))
    prediction_result = db.Column(db.String(100))
    stage_message = db.Column(db.Text)
    sepsis_probability = db.Column(db.Float)

    # Separated input fields
    hr = db.Column(db.Float)
    temp = db.Column(db.Float)
    resp = db.Column(db.Float)
    o2sat = db.Column(db.Float)
    sbp = db.Column(db.Float)
    map_ = db.Column(db.Float)
    dbp = db.Column(db.Float)
    wbc = db.Column(db.Float)
    lactate = db.Column(db.Float)
    creatinine = db.Column(db.Float)
    bun = db.Column(db.Float)
    platelets = db.Column(db.Float)
    glucose = db.Column(db.Float)
    ph = db.Column(db.Float)
    hco3 = db.Column(db.Float)
    paco2 = db.Column(db.Float)
    fio2 = db.Column(db.Float)
    ptt = db.Column(db.Float)
    fibrinogen = db.Column(db.Float)
    age = db.Column(db.Float)
    gender = db.Column(db.Integer)
    sepsislabel = db.Column(db.Integer)

    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


@app.route('/')
@app.route('/first')
def home_page():
    return render_template('first.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = Doctor.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            response = make_response(redirect(url_for('home')))  # Redirect to /home after login
            response.set_cookie('user_id', str(user.id), max_age=60 * 60 * 24 * 10)  # 10 days
            return response
        else:
            flash('Login Unsuccessful. Check email and password.', 'danger')
    return render_template('login.html', form=form)


@app.route('/home')
@login_required
def home():
    user_id = request.cookies.get('user_id')
    doctor = Doctor.query.get(user_id)
    return render_template('home.html', doctor=doctor)


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = request.cookies.get('user_id')
    doctor = Doctor.query.get(user_id)
    return render_template('dashboard.html', doctor=doctor)

@app.route('/patient_entry',methods = ['GET','POST'])
@login_required
def patient_entry():
    print(request.method)
    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        ward = request.form['ward']
        attendee = request.form['attendee']
        address = request.form['address']
        sex = request.form['sex']
        consultant = request.form['consultant']
        admission = request.form['admission']
        blood = request.form['blood']
        timeseries = request.form['timeseries']
        additional = request.form.get('additional', '')  # Optional field

        # Create a new Patient object
        new_patient = Patient(
            name=name,
            age=age,
            phone=phone,
            ward=ward,
            attendee=attendee,
            address=address,
            sex=sex,
            consultant=consultant,
            admission_date=admission,
            blood_sample_report=blood,
            timeseries_samples=timeseries,
            additional_observations=additional
        )
        # Save to database
        try:
            db.session.add(new_patient)
            db.session.commit()
            flash('Patient details submitted successfully!', 'success')
            return redirect(url_for('patient_entry'))  # Redirect to the same page or another route
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')
    else:
        return render_template('patient_entry.html')


    return render_template('patient_entry.html')



@app.route('/doctor-profile', methods=['GET', 'POST'])
@login_required
def doctor_profile():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if check_password_hash(current_user.password, form.old_password.data):
            current_user.password = generate_password_hash(form.new_password.data)
            db.session.commit()
            flash("Password updated successfully!", "success")
        else:
            flash("Old password is incorrect.", "danger")
        return redirect(url_for('doctor_profile'))

    # Dummy value for patients registered, replace with real count
    patients_registered = 35

    return render_template('doctor_page.html', doctor=current_user, form=form,
                           patients_registered=patients_registered)


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # Check for duplicates
        existing_email = Doctor.query.filter_by(email=form.email.data).first()
        existing_phone = Doctor.query.filter_by(phone_number=form.phone_number.data).first()

        if existing_email:
            flash('Email already registered.', 'danger')
        elif existing_phone:
            flash('Phone number already registered.', 'danger')
        else:
            # Save to database
            hashed_password = generate_password_hash(form.password.data)
            new_doctor = Doctor(
                name = form.name.data,
                email=form.email.data,
                department=form.department.data,
                phone_number=form.phone_number.data,
                password_hash=hashed_password
            )
            db.session.add(new_doctor)
            db.session.commit()
            flash("Registration successful!", "success")
            return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/sepsis_predict', methods=['GET', 'POST'])
@login_required
def sepsis_predict():
    if request.method == 'POST':
        patient_name = request.form.get('Patient Name')
        user_id = request.cookies.get('user_id')

        input_data = {
            'HR': float(request.form['HR']),
            'Temp': float(request.form['Temp']),
            'Resp': float(request.form['Resp']),
            'O2Sat': float(request.form['O2Sat']),
            'SBP': float(request.form['SBP']),
            'MAP': float(request.form['MAP']),
            'DBP': float(request.form['DBP']),
            'WBC': float(request.form['WBC']),
            'Lactate': float(request.form['Lactate']),
            'Creatinine': float(request.form['Creatinine']),
            'BUN': float(request.form['BUN']),
            'Platelets': float(request.form['Platelets']),
            'Glucose': float(request.form['Glucose']),
            'pH': float(request.form['pH']),
            'HCO3': float(request.form['HCO3']),
            'PaCO2': float(request.form['PaCO2']),
            'FiO2': float(request.form['FiO2']),
            'PTT': float(request.form['PTT']),
            'Fibrinogen': float(request.form['Fibrinogen']),
            'Age': float(request.form['Age']),
            'Gender': int(request.form['Gender']),
        }

        features = list(input_data.keys())
        input_df = pd.DataFrame([input_data], columns=features)
        scaled_input = scaler.transform(input_df[features])
        prediction = model.predict(scaled_input)
        sepsis_probability = float(prediction[0][0])
        predicted_label = 1 if sepsis_probability > 0.5 else 0

        if predicted_label == 1:
            if sepsis_probability > 0.85:
                stage = 'Stage 3: Septic Shock'
                stage_details = STAGE_3_MESSAGE
            elif sepsis_probability > 0.7:
                stage = 'Stage 2: Severe Sepsis'
                stage_details = STAGE_2_MESSAGE
            else:
                stage = 'Stage 1: Initial Sepsis'
                stage_details = STAGE_1_MESSAGE

            return_value = f"High chances of Sepsis — {stage}"
        else:
            stage = 'No Sepsis'
            stage_details = "The patient's condition does not currently indicate sepsis."
            return_value = "No Sepsis Detected"

        # Save to database (SQLAlchemy model)
        record = SepsisPrediction(
            user_id=user_id,
            patient_name=patient_name,
            prediction_result=return_value,
            stage_message=stage_details,
            sepsis_probability=sepsis_probability,
            hr=input_data['HR'],
            temp=input_data['Temp'],
            resp=input_data['Resp'],
            o2sat=input_data['O2Sat'],
            sbp=input_data['SBP'],
            map_=input_data['MAP'],
            dbp=input_data['DBP'],
            wbc=input_data['WBC'],
            lactate=input_data['Lactate'],
            creatinine=input_data['Creatinine'],
            bun=input_data['BUN'],
            platelets=input_data['Platelets'],
            glucose=input_data['Glucose'],
            ph=input_data['pH'],
            hco3=input_data['HCO3'],
            paco2=input_data['PaCO2'],
            fio2=input_data['FiO2'],
            ptt=input_data['PTT'],
            fibrinogen=input_data['Fibrinogen'],
            age=input_data['Age'],
            gender=input_data['Gender'],
            sepsislabel=predicted_label
        )
        db.session.add(record)
        db.session.commit()

        return render_template('sepsis_prediction.html', label=return_value, stage_details=stage_details)

    return render_template('sepsis_prediction.html', probability=None, label=None, stage_details=None)


@app.route('/download_patient_report', methods=['GET'])
@login_required
def download_patient_report():
    patients = Patient.query.all()
    return render_template('download_patient_report.html',patients=patients)



@app.route('/download_report/<string:patient_name>')
@login_required
def download_report(patient_name):
    patient_result = SepsisPrediction.query.filter_by(patient_name=patient_name).first_or_404()

    logo_abs_path = os.path.abspath('static/logo1.jpg').replace('\\', '/')

    rendered_html = render_template('patient_pdf.html', patient=patient_result, logo_path=logo_abs_path)

    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

    options = {
        'enable-local-file-access': None
    }

    pdf_data = pdfkit.from_string(rendered_html, False, configuration=config, options=options)

    return send_file(
        io.BytesIO(pdf_data),
        download_name=f'Patient_{patient_result.patient_name}_Report.pdf',
        mimetype='application/pdf',
        as_attachment=True
    )



@app.route('/patient_history')
@login_required
def patient_history():
    # Fetch all patients from the database
    patients = Patient.query.all()
    return render_template('patient_history.html', patients=patients)

@app.route('/doctor_profile')
@login_required
def all_doctors():
    user_id = request.cookies.get('user_id')
    doctor = Doctor.query.get(user_id)
    if doctor is None:
        return "Doctor not found", 404
    return render_template('doctor_profile.html', doctor=doctor)


@app.route('/logout', methods=['POST'])
def logout():
    response = redirect(url_for('login'))
    response.delete_cookie('user_id')
    return response

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
