# 🫁 PneumoShield: Pneumonia Detection API using Flask & Deep Learning

PneumoShield is an AI-powered Flask API that detects **pneumonia** from chest X-ray images using pre-trained convolutional neural networks. Built as part of a semester mini-project, it integrates machine learning with a modern MERN/Next.js frontend to provide real-time diagnostic assistance.

---

## 🚀 Features

- 🔬 Predicts pneumonia from chest X-ray images  
- 🧠 Supports two deep learning models:  
  - Custom CNN (`cnn_best.h5`)  
  - Pretrained VGG19 (`vgg19_best.h5`)  
- 📁 Accepts image uploads (JPEG/PNG) for prediction  
- 🔄 Returns prediction results via JSON response  
- 🌐 Ready to integrate with any frontend (React/Next.js/Mobile)  
- 🛠️ Built using **Flask**, **TensorFlow/Keras**, **NumPy**, **Pillow**, **CORS**

---

## 📂 Directory Structure

<pre>
pneumonia-api/
├── models/
│   ├── cnn_best.h5             # Trained CNN model
│   └── vgg19_best.h5           # Pretrained VGG19 model
├── app.py                      # Main Flask app
├── predict.py                  # ML prediction logic
├── requirements.txt            # All dependencies
└── README.md                   # You're here
</pre>

---

## 🧪 How It Works

1. Upload a chest X-ray image to the `/predict` endpoint.  
2. Image is preprocessed and passed through both CNN and VGG19 models.  
3. The API returns predictions from both models:  
   - `"Pneumonia"` or `"Normal"`  
   - Probability scores (confidence)

---


