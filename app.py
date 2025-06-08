import os
from datetime import datetime
from flask_mail import Mail, Message
from pyresparser import ResumeParser
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy # Assuming you kept the DB code
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import torch

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

DB_USERNAME = 'root'
DB_PASSWORD = '****' #  CHANGE THIS 
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'resume_screening_db'
app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587 # Port for TLS
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

app.config['MAIL_USERNAME'] = '*******'
app.config['MAIL_PASSWORD'] = '********' # REPLACE THIS with the App Password


app.config['MAIL_DEFAULT_SENDER'] = ('Resume Screening App', app.config['MAIL_USERNAME']) # Customize sender name



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_very_secret_key' 
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
mail=Mail(app)







#Database Models
class Job(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    results = db.relationship('ScreeningResult', backref='job', lazy=True, cascade="all, delete-orphan")
    

class ScreeningResult(db.Model):
    __tablename__ = 'screening_results'
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=True) # Store extracted name
    email = db.Column(db.String(255), nullable=True)
    score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# Load Sentence Transformer Model 
print("Loading Sentence Transformer model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("Loading spaCy NLP model (en_core_web_sm)...")
try:
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None # Set nlp to None if model loading fails
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None    


# --- Helper Functions (Existing) ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Parse pdf
def parse_pdf(file_stream):
    """Extracts text from a PDF file stream."""
    text = ""
    filename_for_error = getattr(file_stream, 'filename', 'N/A')
    try:
        reader = PyPDF2.PdfReader(file_stream)
        if reader.is_encrypted:
             flash(f"Skipping password-protected PDF: {filename_for_error}", "warning")
             return None
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except PyPDF2.errors.PdfReadError as pe:
        print(f"Error reading PDF (PdfReadError - {filename_for_error}): {pe}")
        flash(f"Error reading PDF file (possibly corrupted): {filename_for_error}. Skipping.", "error")
        return None
    except Exception as e:
        print(f"General Error reading PDF ({filename_for_error}): {e}")
        flash(f"Error reading PDF file: {filename_for_error}. Skipping.", "error")
        return None
    return text.strip()


def calculate_similarity(job_description, resume_text):
    """Calculates cosine similarity between JD and resume using Sentence Transformers."""
    if not resume_text or not job_description:
        return 0.0
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model.to(device) # Ensure model is on correct device if necessary
        jd_embedding = model.encode(job_description, convert_to_tensor=True, device=device)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True, device=device)
        cosine_scores = util.cos_sim(jd_embedding, resume_embedding)
        return cosine_scores.item()
    except Exception as e:
        print(f"Error during similarity calculation: {e}")
        flash("An error occurred during similarity calculation.", "error")
        return 0.0
    

def extract_info_with_pyresparser(file_path):
    """Extracts info using pyresparser library."""
    try:
        parser = ResumeParser(file_path) # Pass the file path
        data = parser.get_extracted_data()
        return {
            "name": data.get('name'),
            "email": data.get('email'),
            "mobile_number": data.get('mobile_number'), # Optional extra field
            "skills": data.get('skills'),              # Optional extra field
            "raw_text": parser.get_extracted_data().get('resume_text', '') # Get text if needed later
        }
    except Exception as e:
        print(f"Error using pyresparser on {file_path}: {e}")
        return { # Return None or default values on error
            "name": None,
            "email": None,
            "mobile_number": None,
            "skills": None,
            "raw_text": None
        }

# Flask routes
@app.route('/', methods=['GET'])
def index():
    """Renders the main page for recruiter uploads."""
    return render_template('index.html', results=None, job_description=None) # Assuming index.html is recruiter view

# Recruiter Route 2: Process Resumes
@app.route('/process', methods=['POST'])
def process_resumes():
    """Handles file uploads, processes resumes, saves results to DB, and shows results."""
    # --- Basic Input Validation ---
    if 'job_description' not in request.form or not request.form['job_description']:
        flash('Job Description is required!', 'error')
        return redirect(url_for('index')) # Redirect back to recruiter view

    resume_files = request.files.getlist('resumes')
    # Check if list is empty or first file is empty placeholder
    if not resume_files or not resume_files[0].filename:
        flash('No resume files selected!', 'error')
        return redirect(url_for('index')) # Redirect back to recruiter view

    job_description_text = request.form['job_description'].strip()

    # --- Create Job Entry and Get ID ---
    # Define current_job_id *outside* the try block initially perhaps? No, better inside.
    try:
        new_job = Job(description=job_description_text)
        db.session.add(new_job)
        # Flush the session to get the ID assigned to new_job by the database
        # *before* we commit the transaction. This is crucial.
        db.session.flush()
        # Assign the ID *only if flush was successful*
        current_job_id = new_job.id
        print(f"Created Job entry with ID: {current_job_id}")

    except Exception as e:
        db.session.rollback() # Rollback the failed attempt to add the job
        print(f"Error creating Job entry in MySQL: {e}")
        flash("Database error: Could not save job description. Check DB connection and credentials.", "error")
        # Crucially, exit the function here if the Job couldn't be created
        return redirect(url_for('index')) # Redirect back to recruiter view

    # --- If Job Creation Succeeded, Proceed with Resume Processing ---
    # This part ONLY runs if the 'try' block above was successful and current_job_id is defined.

    results_to_save = []
    processed_files = 0
    error_files = 0
    temp_dir = app.config.get('UPLOAD_FOLDER', 'uploads') # Use upload folder for temp files
    os.makedirs(temp_dir, exist_ok=True) #

    for file in resume_files:
        if file.filename == '':
            continue

        filename = file.filename # Use secure_filename in production
        score_value = None
        status_text = "Unknown Error"
        extracted_name = None
        extracted_email = None
        resume_text_for_similarity = None # Need text separately for similarity calc

        # --- Save file temporarily ---
        # Create a unique temporary filename if processing many concurrently
        temp_file_path = os.path.join(temp_dir, f"temp_{filename}")
        try:
            file.save(temp_file_path) # Save the uploaded file object
        except Exception as e:
            print(f"Error saving temp file {filename}: {e}")
            flash(f"Could not save uploaded file {filename}.", "error")
            error_files += 1
            # Optionally create a result entry with 'Save Error' status
            results_to_save.append(ScreeningResult(
                 job_id=current_job_id, filename=filename, name=None, email=None,
                 score=None, status="Save Error"
            ))
            continue # Skip to next file

        # --- Process the saved file ---
        if allowed_file(filename):
            print(f"Processing file: {temp_file_path}")

            # 1. <<< Use pyresparser for extraction >>>
            extracted_data = extract_info_with_pyresparser(temp_file_path)
            extracted_name = extracted_data.get("name")
            extracted_email = extracted_data.get("email")
            resume_text_for_similarity = extracted_data.get("raw_text") # Get text from parser if possible

            # If pyresparser failed to get text, fallback to old method
            if not resume_text_for_similarity:
                 print("pyresparser did not return text, falling back to manual parsing for similarity.")
                 try:
                    # Need to re-open the saved file to get a stream for old parsers
                    with open(temp_file_path, 'rb') as file_stream:
                        if filename.lower().endswith('.pdf'):
                            resume_text_for_similarity = parse_pdf(file_stream)
                        elif filename.lower().endswith('.docx'):
                             resume_text_for_similarity = parse_docx(file_stream)
                 except Exception as parse_err:
                      print(f"Fallback text parsing failed for {filename}: {parse_err}")


            # 2. Calculate Similarity (if text was extracted)
            if resume_text_for_similarity:
                if not resume_text_for_similarity.strip():
                     print(f"Warning: Empty text extracted from {filename}")
                     status_text = "Empty Content"
                     error_files += 1
                else:
                    similarity_score = calculate_similarity(job_description_text, resume_text_for_similarity)
                    score_value = round(similarity_score * 100, 2)
                    status_text = "Processed"
                    processed_files += 1
            else:
                # If text extraction failed completely
                status_text = "Error Parsing"
                error_files += 1
                if not extracted_name and not extracted_email: # Check if pyresparser got anything
                     flash(f"Failed to extract text and contact info from {filename}.", "error")

        else: # Invalid file type
            flash(f"File type not allowed: {filename}. Allowed: {ALLOWED_EXTENSIONS}", "warning")
            status_text = "Invalid Format"
            error_files += 1

        # Create ScreeningResult object
        result_entry = ScreeningResult(
            job_id=current_job_id,
            filename=filename,
            name=extracted_name,
            email=extracted_email,
            score=score_value,
            status=status_text
        )
        results_to_save.append(result_entry)

        # --- Clean up the temporary file ---
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            print(f"Error deleting temp file {temp_file_path}: {e}")

    # --- Save results to Database ---
    if results_to_save:
        try:
            db.session.add_all(results_to_save)
            db.session.commit() # Commit job and all results together
            print(f"Successfully saved {len(results_to_save)} screening results for Job ID {current_job_id} to MySQL.")
        except Exception as e:
            db.session.rollback()
            print(f"Error saving screening results to MySQL DB: {e}")
            flash("Database error: Could not save screening results. Check DB connection.", "error")
            # Redirect even if saving results fails, Job entry might still exist (or not)
            return redirect(url_for('index')) # Redirect back to recruiter view
    else:
         # If no valid files were processed, commit the Job entry anyway
         try:
            db.session.commit() # Commits the Job added earlier
            print(f"Committed Job ID {current_job_id} with no associated results.")
         except Exception as e:
             db.session.rollback()
             print(f"Error committing Job entry when no files processed: {e}")
             flash("Database error occurred.", "error")
             return redirect(url_for('index')) # Redirect back to recruiter view

    # Check if any actual processing happened, inform user
    if processed_files == 0 and error_files == 0: # Check if truly nothing was submitted or processed
        flash('No valid resume files were found or processed.', 'warning')
        # Don't necessarily redirect here, show the empty results for the created job? Or redirect?
        # Let's proceed to show the (empty) results table for consistency
        # return redirect(url_for('index'))

   
    try:
        all_results_for_job = ScreeningResult.query.filter_by(job_id=current_job_id)\
                                               .order_by(ScreeningResult.score.is_(None).asc(), ScreeningResult.score.desc())\
                                               .all()
    except Exception as e:
        print(f"Error querying results for Job ID {current_job_id}: {e}")
        flash("Database error: Could not retrieve screening results.", "error")
        all_results_for_job = [] # Assign empty list on error


    # Convert DB objects to simple dicts for template
    results_for_template = [
        {'filename': r.filename,
         'name': r.name if r.name else 'N/A',         # Add name
         'email': r.email if r.email else 'N/A', 
         'score': r.score if r.score is not None else 'N/A',
         'status': r.status}
        for r in all_results_for_job
    ]

    emails_sent_count = 0
    emails_failed_count = 0
    top_candidates_to_email = []

    # Filter for processed candidates with valid email and score
    valid_candidates = [
        r for r in all_results_for_job if r.status == 'Processed' and r.email and r.score is not None
    ]

    # Sort again just to be sure (already sorted by query, but explicit sort is fine)
    valid_candidates.sort(key=lambda x: x.score, reverse=True)

    # Get top 5 (or fewer if less than 5 valid candidates)
    top_candidates_to_email = valid_candidates[:5]

    if top_candidates_to_email:
        print(f"Attempting to email top {len(top_candidates_to_email)} candidates...")
        for candidate in top_candidates_to_email:
            recipient_email = candidate.email
            recipient_name = candidate.name if candidate.name else "Candidate" # Use name if available
            score = candidate.score

            # --- Construct Email ---
            subject = f"Update on Your Resume Submission - Score: {score:.2f}%" # Example subject
            # Define email body (can use Flask render_template for HTML emails later)
            body = f"""Dear {recipient_name},

Thank you for submitting your resume.

We have reviewed your resume against the job description using our screening tool, and it received a relevance score of {score:.2f}%.

This score indicates a potential good match based on semantic similarity and keyword analysis. We encourage candidates with high scores to proceed with the application if they haven't already or expect further communication if applicable based on the specific job posting.

[Optional: Add link to job posting or next steps instructions here]

Best regards,
[Your Company/Recruiting Team Name]
"""
            # Create the message object
            # Use sender from config: app.config['MAIL_DEFAULT_SENDER'] or just app.config['MAIL_USERNAME']
            msg = Message(subject=subject,
                          recipients=[recipient_email],
                          body=body,
                          sender=app.config['MAIL_DEFAULT_SENDER']) # Use sender from config

            # --- Send Email ---
            try:
                mail.send(msg)
                print(f"Email successfully sent to {recipient_email}")
                emails_sent_count += 1
            except Exception as e:
                print(f"!!!!!!!! FAILED to send email to {recipient_email}: {e} !!!!!!!!")
                emails_failed_count += 1
                # Optional: Flash a specific warning for failed emails
                flash(f"Warning: Failed to send email to {recipient_email}. Please check configuration/credentials.", "warning")

    # --- Update Flash Message ---
    final_flash_message = f"Processed {processed_files} valid resume(s). Encountered issues with {error_files} file(s). Results saved."
    if emails_sent_count > 0:
        final_flash_message += f" Attempted to send emails to top {len(top_candidates_to_email)} candidates ({emails_sent_count} sent, {emails_failed_count} failed)."
    elif top_candidates_to_email: # Attempted but all failed
         final_flash_message += f" Attempted to send emails to top {len(top_candidates_to_email)} candidates, but all failed. Check email configuration."
    else:
         final_flash_message += " No candidates met criteria for automatic email notification."


    flash(final_flash_message, "info")
    # --- END EMAIL SENDING LOGIC ---

    # Render the RECRUITER view template
    return render_template('index.html', results=results_for_template, job_description=job_description_text)

# --- Function to Create Database Tables (Keep if using DB) ---
def create_db_tables():
    """Creates database tables based on models if they don't exist."""
    print("Attempting to create database tables in MySQL...")
    try:
        with app.app_context():
            db.create_all()
        print(f"Tables created successfully in database '{DB_NAME}' (if they didn't exist).")
    except Exception as e:
        print(f"Error creating database tables in MySQL: {e}")
        print("Please ensure MySQL server is running, the database exists, and connection details are correct.")
    
# --- Run the App ---
if __name__ == '__main__':
    create_db_tables() # Create tables if they don't exist
    app.run(debug=True) # debug=True for development