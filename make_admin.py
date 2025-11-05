from app import db, User

# 🧑‍🌾 Change this to your email address that you used to register
email = "virgilliday53@gmail.com"

user = User.query.filter_by(email=email).first()
if user:
    user.is_admin = True
    db.session.commit()
    print(f"✅ User '{user.username}' is now an ADMIN.")
else:
    print("❌ User not found. Please register first.")