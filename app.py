from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from utils.preprocessor import preprocess_image
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = os.path.join('models', 'cnn_best_fixed.keras')  # ✅ Updated model path
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif', 'tiff', 'dcm'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable (lazy loading)
model = None
model_load_error = None  # ✅ Store error message if model fails to load

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pneumonia_model():
    """Load the pneumonia detection model once"""
    global model, model_load_error
    if model is None:
        try:
            print("🔄 Checking model file...")
            if not os.path.exists(MODEL_PATH):
                model_load_error = f"❌ Model file not found: {MODEL_PATH}"
                print(model_load_error)
                return None
            
            print("🔄 Loading pneumonia model...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully!")
            model_load_error = None  # Clear error on successful load
        except Exception as e:
            model_load_error = f"❌ Error loading model: {e}"
            print(model_load_error)
            model = None
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_status": "Loaded" if model is not None else "Not Loaded",
        "model_error": model_load_error if model_load_error else "None",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    """Home route displaying available routes"""
    return """
    <h1>You are at the home route. Everything is working perfectly!</h1>
    <h2>Available Routes:</h2>
    <ul>
        <li><strong>GET /health</strong> - Check server and model health</li>
        <li><strong>POST /predict</strong> - Upload an X-ray image for pneumonia detection</li>
        <li><strong>POST /cleanup</strong> - Delete old uploaded images</li>
    </ul>
    """, 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict pneumonia from chest X-ray images"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty file provided"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        # Generate a secure filename
        original_filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{original_filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the file
        file.save(file_path)
        print(f"✅ Image saved to {file_path}")

        # Process the image
        img = preprocess_image(file_path)

        # Ensure model is loaded
        global model
        if model is None:
            load_pneumonia_model()

        if model is None:
            return jsonify({"error": "Model failed to load", "model_error": model_load_error}), 500

        # Make prediction
        prediction = model.predict(img)
        prediction_value = float(prediction[0][0])

        # Determine classification
        pneumonia_prob = prediction_value  # Always direct output
        normal_prob = 1 - pneumonia_prob  # Complementary probability
        prediction_label = "Pneumonia" if pneumonia_prob > 0.5 else "Normal"

        # Return prediction result (without exposing full file path)
        result = {
            "prediction": prediction_label,
            "confidence": round(float(max(pneumonia_prob, normal_prob)), 4),
            "normal_probability": round(float(normal_prob), 4),
            "pneumonia_probability": round(float(pneumonia_prob), 4),
            "filename": filename,
            "timestamp": datetime.datetime.now().isoformat()
        }

        print(f"📌 Prediction Result: {result}")  # ✅ Added log
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error in prediction: {e}")  # ✅ Debugging log
        return jsonify({
            "error": str(e),
            "message": "An error occurred while processing the image"
        }), 500

@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    """Retrieve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/cleanup', methods=['POST'])
def cleanup_old_images():
    """Delete old uploaded images"""
    try:
        max_age_hours = request.json.get('max_age_hours', 24)
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
        deleted_count = 0

        if not os.path.exists(UPLOAD_FOLDER):  # ✅ Prevents errors
            return jsonify({"error": "Upload folder does not exist"}), 500

        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if mod_time < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1

        print(f"🗑️ Cleanup done! {deleted_count} files deleted.")  # ✅ Log cleanup
        return jsonify({
            "success": True,
            "deleted_count": deleted_count,
            "cutoff_time": cutoff_time.isoformat()
        })

    except Exception as e:
        print(f"❌ Cleanup error: {e}")  # ✅ Debugging log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    load_pneumonia_model()  # Load the model at startup
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
