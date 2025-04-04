import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.mixed_precision import Policy  # ✅ Correct import
from keras.utils import custom_object_scope

# Define paths
OLD_MODEL_PATH = "models/cnn_best.h5"
INTERMEDIATE_MODEL_PATH = "models/cnn_best_fixed.h5"
NEW_MODEL_PATH = "models/cnn_best_fixed.keras"

print("🔄 Loading model with `custom_object_scope` workaround...")

try:
    # ✅ Define custom objects for safe loading
    custom_objects = {
        "InputLayer": Input,
        "DTypePolicy": Policy  # ✅ Fix dtype policy issue
    }

    # ✅ Load model with custom objects
    with custom_object_scope(custom_objects):
        old_model = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False)

    # ✅ Save as HDF5 format first (fix compatibility)
    print("💾 Saving intermediate HDF5 model...")
    old_model.save(INTERMEDIATE_MODEL_PATH, save_format="h5")

    # ✅ Reload from HDF5 (ensures proper format)
    print("🔄 Reloading from HDF5 model...")
    fixed_model = tf.keras.models.load_model(INTERMEDIATE_MODEL_PATH, compile=False)

    # ✅ Save final `.keras` model
    print("💾 Saving final .keras model...")
    fixed_model.save(NEW_MODEL_PATH, save_format="keras")

    print("✅ Model converted successfully!")

except Exception as e:
    print(f"❌ Error during conversion: {e}")
