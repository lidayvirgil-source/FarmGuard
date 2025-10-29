from tensorflow.keras.models import load_model

# Load your existing .h5 model
model = load_model("tobacco_mobilenetv2.h5", compile=False)

# Save it in the new .keras format (modern TensorFlow format)
model.save("tobacco_mobilenetv2.keras")

print("✅ Model converted successfully! Saved as tobacco_mobilenetv2.keras")
