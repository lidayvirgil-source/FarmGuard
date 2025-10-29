import os
import datetime
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------------------
# 🏗️ App Configuration
# -------------------------------------
app = Flask(__name__)
app.secret_key = "farmguard_secret_key"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///farmguard.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# -------------------------------------
# 🧩 Models
# -------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    histories = db.relationship("History", backref="user", lazy=True)


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    prediction = db.Column(db.String(120))
    recommendation = db.Column(db.String(500))
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


# -------------------------------------
# 🔒 Login Manager
# -------------------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# -------------------------------------
# 🧠 Load ML Model
# -------------------------------------
MODEL_PATH = "tobacco_mobilenetv2.keras"
model = load_model(MODEL_PATH)

CLASS_LABELS = [
    "Anthracnose",
    "Brown_Spot",
    "Frog_eye_Leaf_Spot",
    "Tobacco_Mosaic_Virus",
    "Wildfire",
    "Healthy"
]

# -------------------------------------
# 🧮 Helper Functions
# -------------------------------------
def is_blurry_or_dark(image_path, blur_thresh=100, dark_thresh=50):
    """Check if image is too blurry or dark."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    if blur < blur_thresh or brightness < dark_thresh:
        return True
    return False


def is_likely_leaf(image_path):
    """Basic leaf detection based on green color percentage."""
    img = cv2.imread(image_path)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = (cv2.countNonZero(mask) / (img.size / 3)) * 100
    return green_ratio > 5  # at least 5% green


def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    label_index = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    label = CLASS_LABELS[label_index]

    return label, confidence


def get_recommendation(label):
    recommendations = {
        "Anthracnose": "Remove infected leaves and apply copper fungicides.",
        "Brown_Spot": "Improve airflow, avoid moisture, use mancozeb fungicide.",
        "Frog_eye_Leaf_Spot": "Rotate crops, remove debris, apply early fungicides.",
        "Tobacco_Mosaic_Virus": "Remove infected plants, sanitize tools.",
        "Wildfire": "Use copper bactericide, ensure proper spacing.",
        "Healthy": "Plant is healthy — maintain care routine.",
    }
    return recommendations.get(label, "No recommendation available.")


# -------------------------------------
# 🌐 Routes
# -------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------ Registration ------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# ------------------ Login ------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Welcome back, " + user.username + "!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Try again.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


# ------------------ Logout ------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ------------------ Dashboard ------------------
@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # --- 🧠 Validate image before prediction ---
            if is_blurry_or_dark(filepath):
                flash("⚠️ Image is too blurry or dark. Please upload a clearer photo.", "warning")
                os.remove(filepath)
                return redirect(url_for("dashboard"))

            if not is_likely_leaf(filepath):
                flash("🚫 The uploaded image doesn't look like a tobacco leaf. Please upload a valid leaf image.", "danger")
                os.remove(filepath)
                return redirect(url_for("dashboard"))

            # --- Prediction ---
            label, confidence = predict_disease(filepath)

            if confidence < 50:
                flash("⚠️ The uploaded image doesn't seem to be a tobacco leaf. Please try again with a clearer leaf photo.", "warning")
                os.remove(filepath)
                return redirect(url_for("dashboard"))

            recommendation = get_recommendation(label)

            history = History(
                filename=filename,
                prediction=f"{label} ({confidence:.2f}%)",
                recommendation=recommendation,
                user_id=current_user.id,
            )
            db.session.add(history)
            db.session.commit()

            return render_template("dashboard.html", label=label, confidence=confidence,
                                   recommendation=recommendation, image_file=filename)

    histories = History.query.filter_by(user_id=current_user.id).order_by(History.date.desc()).all()
    return render_template("dashboard.html", histories=histories)


# ------------------ Delete History ------------------
@app.route("/delete_record/<int:record_id>")
@login_required
def delete_record(record_id):
    record = History.query.get_or_404(record_id)
    if record.user_id != current_user.id and not current_user.is_admin:
        flash("You can only delete your own records.", "danger")
        return redirect(url_for("dashboard"))

    db.session.delete(record)
    db.session.commit()
    flash("Record deleted successfully.", "success")
    return redirect(url_for("dashboard"))


# ------------------ Admin Dashboard ------------------
@app.route("/admin")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("Access denied. Admins only!", "danger")
        return redirect(url_for("dashboard"))

    records = History.query.all()
    users = User.query.all()
    return render_template("admin.html", records=records, users=users)


# ------------------ Admin Delete User ------------------
@app.route("/admin/delete_user/<int:user_id>")
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash("Access denied. Admins only!", "danger")
        return redirect(url_for("dashboard"))

    if current_user.id == user_id:
        flash("You cannot delete your own admin account.", "warning")
        return redirect(url_for("admin_dashboard"))

    user = User.query.get_or_404(user_id)
    History.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()

    flash(f"User '{user.username}' deleted successfully.", "success")
    return redirect(url_for("admin_dashboard"))


# ------------------ Uploaded Image Access ------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -------------------------------------
# 🚀 Run Server
# -------------------------------------
if __name__ == "__main__":
    if not os.path.exists("farmguard.db"):
        with app.app_context():
            db.create_all()
            print("✅ Database created successfully.")
    app.run(debug=True)


# -------------------------------------
# 🧠 Training Improvement Notes
# -------------------------------------
"""
💡 MODEL IMPROVEMENT SUGGESTIONS (for future retraining):

1️⃣ Train for more epochs:
   - Try training the model for 15–20 epochs.
   - This may improve accuracy, but monitor validation loss to avoid overfitting.

2️⃣ Fine-tune MobileNetV2:
   - Unfreeze some of the later convolutional layers of MobileNetV2.
   - Retrain with a smaller learning rate (e.g., 1e-5) to capture tobacco-specific features.

3️⃣ Data Augmentation:
   - Use rotation, zoom, flipping, and brightness adjustments to increase dataset diversity.

4️⃣ Early Stopping and Checkpointing:
   - Use `EarlyStopping` and `ModelCheckpoint` to keep only the best-performing model.

✅ After retraining:
   - Save the model as `.keras` (not .h5) using `model.save("tobacco_mobilenetv2.keras")`.
   - Replace the old file in your project and redeploy.
"""
