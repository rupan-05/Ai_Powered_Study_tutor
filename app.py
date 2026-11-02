from flask import Flask, render_template, request, send_file, abort, url_for, redirect, flash
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from model import predict_assessment_scores

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here' # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///learning_platform.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# --- MODELS ---

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    # Demographic data for AI predictions
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    grade = db.Column(db.Integer)
    parent_occupation = db.Column(db.String(100))

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    study_time = db.Column(db.Float, nullable=False)
    math_score = db.Column(db.Float, nullable=False)
    science_score = db.Column(db.Float, nullable=False)
    english_score = db.Column(db.Float, nullable=False)
    social_score = db.Column(db.Float, nullable=False)
    raw_math_score = db.Column(db.Float, nullable=True)
    raw_science_score = db.Column(db.Float, nullable=True)
    raw_english_score = db.Column(db.Float, nullable=True)
    raw_social_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- PATH CONFIGURATION ---
# This automatically finds the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_MATERIALS_DIR = os.path.join(BASE_DIR, 'study_materials')

# Route for the homepage
@app.route('/')
def index():
    if current_user.is_authenticated:
        # Fetch latest assessment
        latest = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.timestamp.desc()).first()
        return render_template('index.html', latest=latest)
    return render_template('index.html')

import requests
import markdown
import json
import re

# Configure Hugging Face Inference API
# Put your free Hugging Face API token here (starts with hf_...)
HF_API_TOKEN = "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"
API_URL = "https://router.huggingface.co/v1/chat/completions" # Using new stable router
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Setup simple file-based cache
CACHE_DIR = os.path.join(BASE_DIR, 'ai_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(topic, level):
    # Create a safe filename for the cache based on topic and level
    safe_topic = "".join([c if c.isalnum() else "_" for c in topic.lower()])
    return f"{safe_topic}_{level.lower()}.json"

def categorize_topic(topic):
    try:
        prompt = f"Categorize the topic '{topic}' into exactly one of these four categories: Math, Science, English, Social. Return only the category name."
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a precise classifier. Output only one word: Math, Science, English, or Social."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0 # Extreme precision
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            category = result['choices'][0]['message']['content'].strip()
            # Basic cleanup to ensure we match the keys in our dictionary
            for valid in ["Math", "Science", "English", "Social"]:
                if valid.lower() in category.lower():
                    return valid
        return None
    except:
        return None

def generate_notes(topic, level):
    cache_file = os.path.join(CACHE_DIR, get_cache_key(topic, level))
    
    # 1. Try serving from Cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {"status": "success", "content": data.get("html_content", ""), "source": "cache"}
        except Exception as e:
            print(f"Cache read error: {e}") # Silently ignore cache read errors and try API

    # 2. Try calling Hugging Face API
    try:
        # Super-strict prompt for expert tutor with [MATH] tag protection
        prompt = (
            f"Act as an academic tutor. Create structured study notes for {topic} at a {level} level.\n"
            f"MATH RULES:\n"
            f"1. You MUST wrap ONLY mathematical formulas, equations, or scientific notation in [MATH] and [/MATH] tags.\n"
            f"   Example: [MATH]a \\cdot b[/MATH] or [MATH]E = mc^2[/MATH].\n"
            f"2. DO NOT wrap entire questions, sentences, or plain text in these tags.\n"
            f"3. DO NOT use backslashes outside of these tags.\n"
            f"4. Do NOT use any other delimiters like $$ or \\(.\n\n"
            f"Structure:\n"
            f"Overview (Simple explanation)\n"
            f"Core Concepts (Bulleted list)\n"
            f"Why it Matters (Real-world application)\n"
            f"Quick Quiz (3 check questions at the end)."
        )
        
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a precise academic tutor. CRITICAL RULE: You MUST wrap ONLY mathematical symbols, variables, or formulas inside [MATH] and [/MATH] tags. Never wrap plain text or entire questions in these tags. Never use raw backslashes outside these tags. This is the only way the system can render your math."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3 # Lower temperature for better rule following
        }
        
        # Increase timeout for AI generation
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            return None
            
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            generated_text = result['choices'][0]['message']['content']
        else:
            return None
        
        # Convert markdown to HTML first
        html_notes = markdown.markdown(generated_text)
        
        # Post-process to replace [MATH] tags with MathJax delimiters \( \)
        # This protects the LaTeX from being mangled by the markdown parser.
        # We also use a heuristic to catch instances where the AI incorrectly wraps entire English sentences.
        def math_replacer(match):
            content = match.group(1)
            # If the content contains a question mark or many words without math operators, it's likely a sentence
            has_math_symbols = any(c in content for c in ['=', '+', '-', '^', '\\', '/', '_', '*', '<', '>'])
            if '?' in content or (len(content.split()) > 4 and not has_math_symbols):
                return content  # Strip the math tags
            return '\\(' + content + '\\)'

        html_notes = re.sub(r'\[MATH\](.*?)\[/MATH\]', math_replacer, html_notes, flags=re.DOTALL)
        
        # Save to cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({"html_content": html_notes, "topic": topic, "level": level}, f)
        except Exception as e:
            print(f"Cache write error: {e}") 
            
        return {"status": "success", "content": html_notes, "source": "api"}
        
    except Exception as e:
        # Final Clean Fallback for demo: return None to hide the section entirely if the API fails
        return None

# Route for the assessment form
@app.route('/assessment', methods=['GET', 'POST'])
@login_required
def assessment_form():
    if request.method == 'POST':
        try:
            # Collect and validate form data
            student_data = {
                "name": request.form['name'],
                "age": int(request.form['age']),
                "gender": request.form['gender'],
                "grade": int(request.form['grade']),
                "parent_occupation": request.form['parent_occupation'],
                "study_time": float(request.form['study_time']),
                "math_score": float(request.form['math_score']),
                "science_score": float(request.form['science_score']),
                "english_score": float(request.form['english_score']),
                "social_score": float(request.form['social_score'])
            }

            # Predict assessment scores and levels
            results = predict_assessment_scores(student_data)

            # --- Save Demographics to User ---
            current_user.age = student_data["age"]
            current_user.gender = student_data["gender"]
            current_user.grade = student_data["grade"]
            current_user.parent_occupation = student_data["parent_occupation"]

            # --- Save Assessment to Database ---
            new_assessment = Assessment(
                user_id=current_user.id,
                study_time=student_data["study_time"],
                math_score=results.get('Math', {}).get('assessment_score', student_data['math_score']),
                science_score=results.get('Science', {}).get('assessment_score', student_data['science_score']),
                english_score=results.get('English', {}).get('assessment_score', student_data['english_score']),
                social_score=results.get('Social', {}).get('assessment_score', student_data['social_score']),
                raw_math_score=student_data['math_score'],
                raw_science_score=student_data['science_score'],
                raw_english_score=student_data['english_score'],
                raw_social_score=student_data['social_score']
            )
            db.session.add(new_assessment)
            db.session.commit()

            # Redirect to the dashboard
            flash("Assessment completed! Welcome to your personalized dashboard.", "success")
            return redirect(url_for('index'))

        except (ValueError, KeyError) as e:
            return f"Invalid input: {str(e)}. Please check your entries."

    return render_template('assessment_form.html')

@app.route('/study', methods=['POST'])
@login_required
def study():
    topic = request.form.get('topic')
    if not topic:
        flash("Please enter a topic to study.", "warning")
        return redirect(url_for('index'))

    # Fetch latest assessment for AI context
    latest = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.timestamp.desc()).first()
    
    if not latest:
        flash("Please complete your first assessment to unlock study features.", "info")
        return redirect(url_for('update_stats'))

    # Map raw scores to levels for prompt
    def get_level(score):
        if score >= 80: return "Advanced"
        if score >= 60: return "Intermediate"
        return "Beginner"

    student_data = {
        "name": current_user.username,
        "grade": current_user.grade
    }
    
    results = {
        "Math": {"assessment_score": latest.math_score, "level": get_level(latest.math_score)},
        "Science": {"assessment_score": latest.science_score, "level": get_level(latest.science_score)},
        "English": {"assessment_score": latest.english_score, "level": get_level(latest.english_score)},
        "Social": {"assessment_score": latest.social_score, "level": get_level(latest.social_score)}
    }
    
    avg_score = (latest.math_score + latest.science_score + latest.english_score + latest.social_score) / 4
    overall_level = get_level(avg_score)
    
    active_subject = categorize_topic(topic)
    
    if active_subject in results:
        notes_level = results[active_subject]['level']
    else:
        notes_level = overall_level
        
    topic_notes = generate_notes(topic, notes_level)
    
    return render_template('results.html', 
                         topic=topic, 
                         notes_level=notes_level,
                         topic_notes=topic_notes,
                         latest=latest,
                         student=student_data,
                         results=results,
                         overall_level=overall_level,
                         active_subject=active_subject)

@app.route('/update_stats', methods=['GET', 'POST'])
@login_required
def update_stats():
    # If user hasn't done any assessment yet, they won't have demographics
    if not current_user.age:
        flash("Please complete the full assessment form first to set up your profile.", "info")
        return redirect(url_for('assessment_form'))

    if request.method == 'POST':
        try:
            # Update parent occupation if provided
            new_occ = request.form.get('parent_occupation')
            if new_occ:
                current_user.parent_occupation = new_occ

            # Prepare data for prediction model using saved demographics
            student_data = {
                "age": current_user.age,
                "grade": current_user.grade,
                "parent_occupation": current_user.parent_occupation,
                "study_time": float(request.form.get('study_time', 0)),
                "math_score": float(request.form.get('math_score', 0)),
                "science_score": float(request.form.get('science_score', 0)),
                "english_score": float(request.form.get('english_score', 0)),
                "social_score": float(request.form.get('social_score', 0))
            }

            # Run prediction model
            results = predict_assessment_scores(student_data)

            # Save the PREDICTED assessment scores, not just raw marks
            new_assessment = Assessment(
                user_id=current_user.id,
                study_time=student_data["study_time"],
                math_score=results.get('Math', {}).get('assessment_score', student_data['math_score']),
                science_score=results.get('Science', {}).get('assessment_score', student_data['science_score']),
                english_score=results.get('English', {}).get('assessment_score', student_data['english_score']),
                social_score=results.get('Social', {}).get('assessment_score', student_data['social_score']),
                raw_math_score=student_data['math_score'],
                raw_science_score=student_data['science_score'],
                raw_english_score=student_data['english_score'],
                raw_social_score=student_data['social_score']
            )
            db.session.add(new_assessment)
            db.session.commit()
            flash("Proficiency levels updated and predicted by AI!", "success")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Error updating stats: {str(e)}", "danger")

    latest = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.timestamp.desc()).first()
    return render_template('update_stats.html', latest=latest)

# Route for downloading study materials
@app.route('/materials/<subject>/<level>/<grade>')
@login_required
def materials(subject, level, grade):
    try:
        # Ensure grade is parsed correctly
        grade_val = int(grade)

        # Construct the path using the relative directory
        # Matches your folder structure: study_materials/[Subject]/[Level]/Class[Grade].pdf
        pdf_filename = f'Class{grade_val}.pdf'
        pdf_path = os.path.join(STUDY_MATERIALS_DIR, subject, level, pdf_filename)

        # Check if the file exists before attempting to send it
        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True)
        else:
            # Helpful error message if the file is missing
            return f"Error: File not found at {pdf_path}", 404
    
    except ValueError:
        abort(400, description="Invalid grade value. Please enter a valid number.")

# --- AUTH ROUTES ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check username and password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    # Fetch all assessments for the current user, newest first
    assessments = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.timestamp.desc()).all()
    
    # Calculate trends (Status)
    # We compare each assessment with the one immediately following it (the previous one chronologically)
    history_data = []
    for i in range(len(assessments)):
        curr = assessments[i]
        trend = "Neutral" # Default
        
        if i + 1 < len(assessments):
            prev = assessments[i+1]
            # Simple average score comparison for overall trend
            curr_avg = (curr.math_score + curr.science_score + curr.english_score + curr.social_score) / 4
            prev_avg = (prev.math_score + prev.science_score + prev.english_score + prev.social_score) / 4
            
            if curr_avg > prev_avg:
                trend = "Up"
            elif curr_avg < prev_avg:
                trend = "Down"
        
        history_data.append({
            "record": curr,
            "trend": trend
        })
    
    return render_template('profile.html', history=history_data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)