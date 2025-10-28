# 🌿 FarmGuard: Intelligent Tobacco Disease Detection System

FarmGuard is a **machine learning-based web application** that predicts and manages tobacco plant diseases.  
It helps farmers quickly identify leaf diseases and provides actionable treatment recommendations.

---

## 🚀 Features

✅ **User Authentication** – Register, login, and manage your account.  
✅ **Disease Detection** – Upload or capture an image of a tobacco leaf.  
✅ **ML Prediction** – Predicts possible diseases using a trained MobileNetV2 model (`tobacco_mobilenetv2.h5`).  
✅ **Recommendations** – Suggests treatment and management steps.  
✅ **History Tracking** – Users can view previous uploads and results.  
✅ **Admin Panel** – Admins can view and manage all users and predictions.  
✅ **Camera Integration** – Capture leaf images directly from your phone or webcam.  
✅ **Responsive Design** – Works on both mobile and desktop.

---

## 🧠 Supported Disease Classes

- Anthracnose  
- Brown Spot  
- Frog-eye Leaf Spot  
- Tobacco Mosaic Virus  
- Wildfire  
- Healthy

---

## 🧩 System Architecture

**Frontend:** HTML, CSS, Bootstrap  
**Backend:** Flask (Python)  
**Database:** SQLite (`farmguard.db`)  
**Machine Learning Model:** TensorFlow / Keras (MobileNetV2-based)  
**Hosting Platform:** Render or Heroku

---

## ⚙️ Installation Guide

### 🖥️ Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/FarmGuard.git
cd FarmGuard
