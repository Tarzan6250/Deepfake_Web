from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
from PIL import Image
import cv2
import numpy as np
import matlab.engine
import base64
from io import BytesIO
from pymongo import MongoClient
from dotenv import load_dotenv
import bcrypt
from functools import wraps
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# MongoDB setup
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DB_NAME')]
users = db.users
analysis_history = db.analysis_history

# Initialize MATLAB engine
eng = matlab.engine.start_matlab()
model_path = r"C:/Nantha/Projects/deepFake-web-app/xceptioncpunetwork.mat"
model_data = eng.load(model_path, nargout=1)
mat_model = model_data['trainedNetwork_1']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = users.find_one({'username': username})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if users.find_one({'username': username}):
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if users.find_one({'email': email}):
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        user = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.utcnow()
        }
        
        users.insert_one(user)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/upload')
@login_required
def upload():
    return render_template('index.html')

@app.route('/history')
@login_required
def history():
    user_history = db.analysis_history.find({'user_id': session['user_id']}).sort('analyzed_at', -1)
    return render_template('history.html', history=user_history)

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        file_type = file.filename.rsplit('.', 1)[1].lower()
        
        if file_type in ['jpg', 'jpeg', 'png']:
            result = analyze_image(file)
        else:
            result = analyze_video(file)
        
        # Store analysis result in MongoDB
        analysis_record = {
            'user_id': session['user_id'],
            'filename': file.filename,
            'result': result,
            'analyzed_at': datetime.utcnow()
        }
        analysis_history.insert_one(analysis_record)
            
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

def analyze_image(file):
    # Load and process image
    image = Image.open(file)
    image_resized = image.resize((299, 299))
    image_array = np.array(image_resized, dtype=np.uint8)
    matlab_img = matlab.uint8(image_array.tolist())
    
    try:
        # Analyze with MATLAB model
        YPred, probs = eng.classify(mat_model, matlab_img, nargout=2)
        label = str(YPred)
        probs_list = list(probs[0])
        confidence = max(probs_list) if probs_list else 0
        
        return {
            'result': 'Real' if label.lower() == 'real' else 'Fake',
            'confidence': float(confidence * 100),
            'type': 'image'
        }
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        raise

def analyze_video(file):
    temp_path = 'temp_video.mp4'
    try:
        file.save(temp_path)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        frames_processed = 0
        faces_detected = 0
        fake_frames = 0
        real_frames = 0
        frame_skip = 30  # Process every 30th frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_processed += 1
            if frames_processed % frame_skip == 0:
                try:
                    resized_frame = cv2.resize(frame, (299, 299))
                    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    matlab_frame = matlab.uint8(resized_frame_rgb.tolist())
                    
                    YPred, _ = eng.classify(mat_model, matlab_frame, nargout=2)
                    label = str(YPred)
                    
                    faces_detected += 1
                    if label.lower() == "fake":
                        fake_frames += 1
                    else:
                        real_frames += 1

                    # Process at least 10 frames but no more than 60
                    if faces_detected >= 60:
                        break
                except Exception as e:
                    print(f"Error processing frame {frames_processed}: {str(e)}")
                    continue

        cap.release()

        if faces_detected == 0:
            return {'result': 'Error', 'confidence': 0, 'type': 'video', 'error': 'No frames could be analyzed'}

        # Calculate final result
        fake_percentage = (fake_frames / faces_detected) * 100
        real_percentage = (real_frames / faces_detected) * 100
        
        is_fake = fake_frames > real_frames
        confidence = fake_percentage if is_fake else real_percentage

        return {
            'result': 'Fake' if is_fake else 'Real',
            'confidence': confidence,
            'type': 'video',
            'frames_analyzed': faces_detected
        }

    except Exception as e:
        print(f"Error in analyze_video: {str(e)}")
        raise

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)