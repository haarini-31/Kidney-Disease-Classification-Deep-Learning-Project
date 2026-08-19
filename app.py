from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_from_directory
import os
import tempfile
import uuid
from functools import wraps
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from models import db, User, Patient, Scan, AIResult, Review, Report, Encounter
import io
from flask import make_response
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ['FLASK_DEBUG'] = '0'
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Database & Uploads Configuration
basedir = os.path.abspath(os.path.dirname(__file__))

# Bind to environment variables for production flexibility
db_url = os.environ.get('DATABASE_URL')
if db_url:
    # SQLAlchemy 1.4+ compatibility for postgres:// URI scheme
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'renalscan.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-prototype-key-renalscan')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(basedir, 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

db.init_app(app)

with app.app_context():

    db.create_all()
    if not User.query.filter_by(email='doctorA@renalscan.demo').first():
        doctor_a = User(name='Dr. Priya (Doctor A)', email='doctorA@renalscan.demo', role='Doctor')
        doctor_a.set_password('password')
        db.session.add(doctor_a)
    if not User.query.filter_by(email='doctorB@renalscan.demo').first():
        doctor_b = User(name='Dr. Arun (Doctor B)', email='doctorB@renalscan.demo', role='Doctor')
        doctor_b.set_password('password')
        db.session.add(doctor_b)
    if not User.query.filter_by(email='demo@renalscan.com').first():
        demo = User(name='Dr. Demo', email='demo@renalscan.com', role='Doctor')
        demo.set_password('password')
        db.session.add(demo)
    db.session.commit()


try:
    classifier = PredictionPipeline()
except Exception as e:
    print(f"Warning: Model could not be loaded on startup: {e}")
    classifier = None

# ---------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid email or password."

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------------------------------------------------------
# PROTECTED ROUTES
# ---------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    total_patients = Patient.query.count()
    total_analyses = Scan.query.count()
    completed_reports = Report.query.count()
    # Pending reviews: Scans that do NOT have a Review OR have a Review with 'Needs further review'
    pending_reviews = db.session.query(Scan).outerjoin(Review).filter(
        db.or_(Review.id == None, Review.status == 'Needs further review')
    ).count()

    recent_activity = Scan.query.order_by(Scan.scan_date.desc()).limit(5).all()

    return render_template('dashboard.html', 
                         user_name=session.get('user_name'),
                         user_role=session.get('user_role'),
                         total_patients=total_patients,
                         total_analyses=total_analyses,
                         completed_reports=completed_reports,
                         pending_reviews=pending_reviews,
                         recent_activity=recent_activity)

@app.route('/patients', methods=['GET'])
@login_required
def patients():
    search = request.args.get('q', '')
    sort_order = request.args.get('sort', 'newest')
    
    query = Patient.query
    if search:
        query = query.filter(
            db.or_(
                Patient.name.ilike(f'%{search}%'),
                Patient.patient_id.ilike(f'%{search}%')
            )
        )
        
    if sort_order == 'oldest':
        query = query.order_by(Patient.created_at.asc())
    elif sort_order == 'alpha':
        query = query.order_by(Patient.name.asc())
    else:
        query = query.order_by(Patient.created_at.desc())
        
    patients_list = query.all()
    return render_template('patients.html', patients=patients_list, search=search, sort_order=sort_order)

@app.route('/patients/new', methods=['GET', 'POST'])
@login_required
def new_patient():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        # Validation
        if not patient_id or not name:
            flash('Patient ID and Name are required.', 'error')
            return render_template('new_patient.html')
            
        if Patient.query.filter_by(patient_id=patient_id).first():
            flash(f'Patient ID {patient_id} is already registered.', 'error')
            return render_template('new_patient.html')
            
        new_pat = Patient(
            patient_id=patient_id,
            name=name,
            age=int(age) if age and age.isdigit() else None,
            gender=gender
        )
        db.session.add(new_pat)
        db.session.commit()
        
        flash('Patient registered successfully.', 'success')
        return redirect(url_for('patient_profile', id=new_pat.id))
        
    return render_template('new_patient.html')

@app.route('/patient/<int:id>', methods=['GET'])
@login_required
def patient_profile(id):
    patient = Patient.query.get_or_404(id)
    # scans will be accessed via patient.scans in template
    return render_template('patient_profile.html', patient=patient)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    demo_dir = os.path.join(app.root_path, 'static', 'demo_scans')
    if os.path.exists(os.path.join(demo_dir, filename)):
        return send_from_directory(demo_dir, filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/scan/new/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def new_scan(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                # 1. Save Image
                ext = file.filename.rsplit('.', 1)[1].lower()
                unique_filename = f"{uuid.uuid4().hex}.{ext}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # 2. Create Encounter
                study_type = request.form.get('study_type', 'Kidney CT')
                encounter = Encounter(
                    patient_id=patient.id,
                    doctor_id=session['user_id'],
                    visit_type=f"{study_type} Analysis"
                )
                db.session.add(encounter)
                db.session.flush() # get encounter.id
                
                # 3. Save Scan Record
                scan = Scan(
                    patient_id=patient.id, 
                    encounter_id=encounter.id,
                    uploaded_by=session['user_id'],
                    study_type=study_type, 
                    image_path=unique_filename
                )
                db.session.add(scan)
                db.session.commit()
                
                # 4. AI Analysis
                if classifier is None:
                    flash('Prediction model unavailable', 'error')
                    return redirect(url_for('patient_profile', id=patient.id))
                    
                prediction, confidence = classifier.predict(file_path)
                
                # 5. Save AI Result
                ai_res = AIResult(
                    scan_id=scan.id, 
                    prediction=prediction, 
                    confidence=round(confidence, 4),
                    model_name='VGG16 Transfer Learning'
                )
                db.session.add(ai_res)
                db.session.commit()
                
                flash('AI Analysis completed.', 'success')
                return redirect(url_for('scan_result', scan_id=scan.id))
                
            except Exception as e:
                print(f"Upload error: {e}")
                db.session.rollback()
                flash('An error occurred processing the image.', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file format. Allowed: JPG, JPEG, PNG.', 'error')
            return redirect(request.url)

    return render_template('new_analysis.html', patient=patient)

@app.route('/scan/<int:scan_id>/result', methods=['GET'])
@login_required
def scan_result(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    return render_template('result.html', scan=scan)

def role_required(roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_role' not in session or session['user_role'] not in roles:
                flash('Unauthorized: You do not have permission to access this resource.', 'error')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/review/<int:scan_id>', methods=['GET', 'POST'])
@login_required
def review(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    review_record = scan.review
    
    if request.method == 'POST':
        # Check permissions for POST
        if session.get('user_role') not in ['Doctor', 'Radiologist']:
            flash('Unauthorized: Medical Staff cannot submit professional reviews.', 'error')
            return redirect(url_for('patient_profile', id=scan.patient_id))
            
        status = request.form.get('status')
        clinical_notes = request.form.get('clinical_notes')
        clinical_impression = request.form.get('clinical_impression')
        recommendation = request.form.get('recommendation')
        
        if not status:
            flash('Review status is required.', 'error')
            return render_template('review.html', scan=scan, review=review_record)
            
        if review_record:
            # Update existing review
            review_record.status = status
            review_record.clinical_notes = clinical_notes
            review_record.clinical_impression = clinical_impression
            review_record.recommendation = recommendation
            review_record.reviewer_id = session['user_id']
        else:
            # Create new review
            review_record = Review(
                scan_id=scan.id,
                reviewer_id=session['user_id'],
                status=status,
                clinical_notes=clinical_notes,
                clinical_impression=clinical_impression,
                recommendation=recommendation
            )
            db.session.add(review_record)
            
        db.session.commit()
        
        # Ensure a Report record exists
        if not scan.report:
            report_record = Report(scan_id=scan.id, review_id=review_record.id)
            db.session.add(report_record)
            db.session.commit()
            
        flash('Professional review saved successfully.', 'success')
        return redirect(url_for('report', scan_id=scan.id))
        
    return render_template('review.html', scan=scan, review=review_record)

@app.route('/report/<int:scan_id>', methods=['GET'])
@login_required
def report(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    return render_template('report.html', scan=scan)

@app.route('/history', methods=['GET'])
@login_required
def history():
    search = request.args.get('q', '')
    filter_status = request.args.get('status', '')
    filter_ai = request.args.get('ai_class', '')

    query = db.session.query(Scan).join(Patient).outerjoin(AIResult).outerjoin(Review)

    if search:
        query = query.filter(db.or_(
            Patient.name.ilike(f'%{search}%'),
            Patient.patient_id.ilike(f'%{search}%')
        ))
        
    if filter_status:
        if filter_status == 'Pending':
            query = query.filter(Review.id == None)
        else:
            query = query.filter(Review.status == filter_status)
            
    if filter_ai:
        query = query.filter(AIResult.prediction == filter_ai)

    scans = query.order_by(Scan.scan_date.desc()).all()
    
    return render_template('history.html', scans=scans, search=search, filter_status=filter_status, filter_ai=filter_ai)

@app.route('/reports', methods=['GET'])
@login_required
def reports_list():
    search = request.args.get('q', '')
    filter_status = request.args.get('status', '')
    sort_order = request.args.get('sort', 'newest')

    query = db.session.query(Report).join(Scan, Report.scan_id == Scan.id).join(Patient, Scan.patient_id == Patient.id).join(Review, Report.review_id == Review.id)

    if search:
        query = query.filter(db.or_(
            Patient.name.ilike(f'%{search}%'),
            Patient.patient_id.ilike(f'%{search}%')
        ))
    if filter_status:
        query = query.filter(Review.status == filter_status)

    if sort_order == 'oldest':
        query = query.order_by(Report.created_at.asc())
    else:
        query = query.order_by(Report.created_at.desc())

    reports = query.all()
    return render_template('reports.html', reports=reports, search=search, filter_status=filter_status, sort_order=sort_order)

@app.route('/report/<int:scan_id>/pdf', methods=['GET'])
@login_required
def report_pdf(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if not scan.report:
        flash('Report not generated yet.', 'error')
        return redirect(url_for('report', scan_id=scan.id))
        
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "RENALSCAN AI")
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 70, "Kidney Imaging Report")
    
    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    disclaimer = "This report is generated as part of an educational/research prototype. AI-generated classification is an assistive output and does not constitute a medical diagnosis. The result must be reviewed and interpreted by a qualified healthcare professional."
    
    import textwrap
    lines = textwrap.wrap(disclaimer, width=100)
    y = height - 100
    for line in lines:
        c.drawString(50, y, line)
        y -= 12
        
    # Patient Info
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Information")
    c.line(50, y - 5, width - 50, y - 5)
    y -= 25
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Name: {scan.patient.name}")
    c.drawString(300, y, f"Patient ID: {scan.patient.patient_id}")
    y -= 20
    c.drawString(50, y, f"Age: {scan.patient.age or 'N/A'}")
    c.drawString(300, y, f"Gender: {scan.patient.gender or 'N/A'}")
    
    # Scan Info
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Scan Details")
    c.line(50, y - 5, width - 50, y - 5)
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Study Type: {scan.study_type}")
    c.drawString(300, y, f"Scan Date: {scan.scan_date.strftime('%Y-%m-%d %H:%M')}")
    
    # AI Analysis
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "AI Analysis")
    c.line(50, y - 5, width - 50, y - 5)
    y -= 25
    c.setFont("Helvetica", 10)
    if scan.ai_result:
        c.drawString(50, y, f"Classification: {scan.ai_result.prediction}")
        c.drawString(300, y, f"Confidence: {scan.ai_result.confidence * 100:.2f}%")
        y -= 20
        c.drawString(50, y, f"Model: {scan.ai_result.model_name}")
    else:
        c.drawString(50, y, "No AI Analysis available.")
        
    # Professional Review
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Professional Review")
    c.line(50, y - 5, width - 50, y - 5)
    y -= 25
    c.setFont("Helvetica", 10)
    if scan.review:
        c.drawString(50, y, f"Status: {scan.review.status}")
        y -= 20
        c.drawString(50, y, "Clinical Notes:")
        y -= 15
        notes_lines = textwrap.wrap(scan.review.clinical_notes or "None", width=90)
        for line in notes_lines:
            c.drawString(50, y, line)
            y -= 15
        
        y -= 10
        c.drawString(50, y, "Recommendation:")
        y -= 15
        rec_lines = textwrap.wrap(scan.review.recommendation or "None", width=90)
        for line in rec_lines:
            c.drawString(50, y, line)
            y -= 15
            
    else:
        c.drawString(50, y, "Pending Review.")
        y -= 15

    c.line(50, y - 15, width - 50, y - 15)

    c.showPage()
    c.save()
    
    buffer.seek(0)
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=RenalScan_Report_{scan.id}.pdf'
    return response

@app.route('/model', methods=['GET'])
@login_required
def model_info():
    return render_template('model.html')

@app.route('/about', methods=['GET'])
@login_required
def about():
    return render_template('about.html')

# ---------------------------------------------------------
# EXISTING API
# ---------------------------------------------------------

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    if classifier is None:
        return jsonify({'error': 'Model is not trained or loaded yet.'}), 503

    try:
        message = request.get_json(force=True)
        if not message or 'image' not in message:
            return jsonify({'error': 'Invalid request: "image" base64 payload is missing.'}), 400
            
        encoded = message['image']
        if not encoded:
            return jsonify({'error': 'Empty image data.'}), 400

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_filepath = temp_file.name
        temp_file.close()

        try:
            decodeImage(encoded, temp_filepath)
            prediction, confidence = classifier.predict(temp_filepath)
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
