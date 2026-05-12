from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from database import users
import datetime
import os
import subprocess
import atexit
import sys
import base64
import io
import socket

import numpy as np
from PIL import Image
from utils.prediction_service import explain_image, predict_image, preload_pretrained_model

app = Flask(__name__)
# Secure session cookie with a secret key
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_for_dev')
app.config["STREAMLIT_PORT"] = int(os.environ.get("STREAMLIT_PORT", "8501"))


@app.context_processor
def inject_user_profile():
    host = request.host.split(":", 1)[0] if request.host else "localhost"
    streamlit_port = app.config.get("STREAMLIT_PORT", 8501)
    streamlit_scheme = request.headers.get("X-Forwarded-Proto", request.scheme or "http")
    return {
        "current_username": session.get("user"),
        "current_email": session.get("email"),
        "streamlit_embed_url": f"{streamlit_scheme}://{host}:{streamlit_port}",
    }


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _find_streamlit_port(start_port: int = 8501, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        if not _port_in_use(port):
            return port
    return start_port

def _decode_frame_data(frame_data: str) -> Image.Image:
    if "," in frame_data:
        frame_data = frame_data.split(",", 1)[1]
    frame_bytes = base64.b64decode(frame_data)
    return Image.open(io.BytesIO(frame_bytes)).convert("RGB")


def _encode_np_image_to_data_url(rgb_image: np.ndarray) -> str:
    buff = io.BytesIO()
    Image.fromarray(rgb_image).save(buff, format="PNG")
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

@app.route('/')
@app.route('/home')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Here you would typically send an email or store the message in the database.
        # For now, we simulate a successful submission.
        flash('Thank you for contacting us! We will get back to you shortly.', 'success')
        return redirect(url_for('contact'))
        
    return render_template('contact.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')

        # Form validation
        if not username or not email or not password or not confirm:
            flash('All fields are required.', 'error')
            return redirect(url_for('signup'))

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('signup'))
        
        # Check if email is already registered
        if users.find_one({'email': email}):
            flash('Email is already registered. Please log in.', 'error')
            return redirect(url_for('signup'))
        
        # Hash the password prior to storing
        hashed_pw = generate_password_hash(password)
        
        new_user = {
            'username': username,
            'email': email,
            'password': hashed_pw,
            'created_at': datetime.datetime.utcnow()
        }
        
        # Insert user to database
        users.insert_one(new_user)
        flash('Account created successfully. Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash('Email and password are required.', 'error')
            return redirect(url_for('login'))

        # Fetch from DB
        user = users.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']  # Start user session
            session['email'] = user.get('email')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user from session
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Please log in to access the dashboard.', 'error')
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', username=session['user'], user_email=session.get('email', ''))


@app.route('/live-detection')
def live_detection():
    if 'user' not in session:
        flash('Please log in to access live detection.', 'error')
        return redirect(url_for('login'))
    return render_template('live_detection.html', username=session['user'])


@app.route('/webcam-predict', methods=['POST'])
def webcam_predict():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    frame_data = payload.get('frame')
    model_mode = payload.get('model_mode', 'pretrained')

    if not frame_data:
        return jsonify({'error': 'Missing frame data'}), 400

    try:
        image = _decode_frame_data(frame_data)
        pred = predict_image(image, model_mode=model_mode, resize_224=True, return_attentions=False)
        return jsonify({
            'label': pred['label'],
            'confidence': round(pred['confidence'] * 100, 2),
            'model_mode': pred['model_mode'],
            'device': pred.get('device', 'cpu'),
            'fp16_enabled': bool(pred.get('fp16_enabled', False)),
            'top_predictions': [
                {
                    'label': item['label'],
                    'confidence': round(float(item['confidence']) * 100, 2),
                }
                for item in (pred.get('top_predictions') or [])
            ],
        })
    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


@app.route('/webcam-explain', methods=['POST'])
def webcam_explain():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    frame_data = payload.get('frame')
    model_mode = payload.get('model_mode', 'pretrained')

    if not frame_data:
        return jsonify({'error': 'Missing frame data'}), 400

    try:
        image = _decode_frame_data(frame_data)
        exp = explain_image(
            image,
            model_mode=model_mode,
            include_gradcam=True,
            include_attention_map=True,
            include_rollout=True,
            include_llm_explanation=True,
            include_integrated_gradients=True,
            resize_224=True,
        )

        return jsonify({
            'label': exp['label'],
            'confidence': round(exp['confidence'] * 100, 2),
            'model_mode': exp['model_mode'],
            'device': exp.get('device', 'cpu'),
            'fp16_enabled': bool(exp.get('fp16_enabled', False)),
            'explanation': exp.get('explanation'),
            'technical_explanation': exp.get('technical_explanation'),
            'contrastive_explanation': exp.get('contrastive_explanation'),
            'top_predictions': [
                {
                    'label': item['label'],
                    'confidence': round(float(item['confidence']) * 100, 2),
                }
                for item in (exp.get('top_predictions') or [])
            ],
            'gradcam': _encode_np_image_to_data_url(exp["gradcam_overlay"]) if exp["gradcam_overlay"] is not None else None,
            'attention_map': _encode_np_image_to_data_url(exp["attention_map_overlay"]) if exp["attention_map_overlay"] is not None else None,
            'attention_rollout': _encode_np_image_to_data_url(exp["attention_rollout_overlay"]) if exp["attention_rollout_overlay"] is not None else None,
            'integrated_gradients': _encode_np_image_to_data_url(exp["integrated_gradients_overlay"]) if exp["integrated_gradients_overlay"] is not None else None,
        })
    except Exception as exc:
        return jsonify({'error': f'Explanation failed: {exc}'}), 500

if __name__ == '__main__':
    # Preload base ViT model once at server startup.
    preload_pretrained_model()

    streamlit_port = _find_streamlit_port(app.config["STREAMLIT_PORT"])
    app.config["STREAMLIT_PORT"] = streamlit_port
    if streamlit_port != 8501:
        print(f"[Info] Port 8501 is busy; using Streamlit port {streamlit_port} instead.", flush=True)

    # Start streamlit in the background
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", 
        "--server.headless=true",
        "--server.address=0.0.0.0",
        f"--server.port={streamlit_port}",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ])
    
    # Ensure streamlit is killed when flask exits
    atexit.register(lambda p: p.terminate(), streamlit_process)
    
    app.run(host="0.0.0.0", debug=True, port=5000, use_reloader=False)
