# 🫁 PneumoShield: Pneumonia Detection API using Flask & Deep Learning

PneumoShield is an AI-powered Flask API that detects **pneumonia** from chest X-ray images using pre-trained convolutional neural networks. Built as part of a semester mini-project, it integrates machine learning with a modern MERN/Next.js frontend to provide real-time diagnostic assistance.
I have also integrated this Working models on my complete project of pneumonia detection.

Link of working project - https://pneumoshield.vercel.app/

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

## ⚙️ Installation & Setup

Follow these steps to run the Pneumonia Detection Flask API locally:

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/alchemist240/Pneumonia-Detection-API.git
cd Pneumonia-Detection-API
```

### 2. 🧪 Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate        # For Linux/macOS
venv\Scripts\activate           # For Windows
```

### 3. 📦 Install the Dependencies

```bash
pip install -r requirements.txt
```

### 4. ▶️ Run the Flask App

```bash
python app.py
```

Once the server starts, you should see:

```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

---

## 📂 Project Structure

<pre>
Pneumonia-Detection-API/
├── models/                   # Contains saved model files (.h5)
├── utils/                    # Helper scripts for image preprocessing or logic
├── .gitignore
├── README.md
├── app.py                   # Main Flask API app
├── convert_model.py         # Script to convert or manage model formats
├── requirements.txt         # All Python dependencies
├── runtime.txt              # Python version info (for deployment)
</pre>

---

## 🧪 API Functionality

- Accepts chest X-ray images (`.jpg`, `.jpeg`, `.png`) via POST request.
- Predicts whether the input image is:
  - `Pneumonia`
  - `Normal`
- Returns JSON response with prediction & confidence score.
- Easily integrates with any frontend (React, Next.js, mobile apps).

---

## 🖼️ Sample Request (Using cURL)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@sample_xray.jpg"
```

**Response Example:**

```json
{
  "prediction": "Pneumonia",
  "confidence": 0.94
}
```

---

## 🔗 Frontend Integration (Example)

```js
const formData = new FormData();
formData.append("file", selectedImageFile);

const res = await fetch("http://localhost:5000/predict", {
  method: "POST",
  body: formData,
});
const data = await res.json();
```

---

## 🧑‍💻 Author

**Kshitij Hundre**  
ML & MERN Stack Enthusiast  
[GitHub](https://github.com/alchemist240)

---

## ⚠️ Disclaimer

> This project is a learning exercise and **not intended for real medical use**. Always consult certified healthcare professionals for actual diagnosis.
```
   - `"Pneumonia"` or `"Normal"`  
   - Probability scores (confidence)

---


