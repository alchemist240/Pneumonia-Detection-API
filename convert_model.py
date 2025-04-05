import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.mixed_precision import Policy
from keras.utils import custom_object_scope
import numpy as np
import os

# Define paths
OLD_MODEL_PATH = "models/cnn_best.h5"
NEW_MODEL_PATH = "models/cnn_best_fixed.keras"

tf.get_logger().setLevel('INFO')  # ✅ Cleaner TensorFlow logging

print("🚀 Starting model conversion...")

try:
    # ✅ Ensure model file exists
    if not os.path.exists(OLD_MODEL_PATH):
        raise FileNotFoundError(f"❌ Old model file not found: {OLD_MODEL_PATH}")
    
    print("📥 Loading old H5 model with custom objects...")
    custom_objects = {
        "InputLayer": Input,
        "DTypePolicy": Policy  # ✅ Fix dtype policy issue
    }

    with custom_object_scope(custom_objects):
        old_model = load_model(OLD_MODEL_PATH, compile=False)  # ✅ Ensure compilation
    
    print("✅ Model loaded successfully!")

    # ✅ Clone model structure and transfer weights
    print("🔄 Rebuilding model with correct weight transfer...")
    new_model = clone_model(old_model)
    new_model.set_weights(old_model.get_weights())

    # ✅ Verify weight consistency
    print("🔍 Verifying weight match...")
    for w1, w2 in zip(old_model.get_weights(), new_model.get_weights()):
        if not np.allclose(w1, w2, atol=1e-6):
            raise ValueError("❌ Weight mismatch detected! Fix the model saving process.")

    print("✅ Weights verified successfully!")

    # ✅ Save final model in `.keras` format
    print("💾 Saving final `.keras` model...")
    new_model.save(NEW_MODEL_PATH, save_format="keras")
    print(f"🎉 Model successfully saved at: {NEW_MODEL_PATH}")

    # ✅ Verify the final model loads correctly
    print("📥 Reloading final `.keras` model for verification...")
    final_model = load_model(NEW_MODEL_PATH, compile=False)
    print("✅ Final model loaded successfully!")

except Exception as e:
    print(f"❌ Error during conversion: {e}")
