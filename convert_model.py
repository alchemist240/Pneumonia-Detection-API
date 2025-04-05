import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.mixed_precision import Policy  # ✅ Correct import
from keras.utils import custom_object_scope
from tensorflow.keras.models import Model
import os

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

    if not os.path.exists(OLD_MODEL_PATH):
        raise FileNotFoundError(f"❌ Old model file not found: {OLD_MODEL_PATH}")
    
    print("📥 Loading old H5 model...")
    with custom_object_scope(custom_objects):
        old_model = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False)

    print("🔄 Rebuilding the model architecture...")
    inputs = old_model.input
    outputs = old_model.output
    rebuilt_model = Model(inputs=inputs, outputs=outputs, name="rebuilt_model")

    rebuilt_model.set_weights(old_model.get_weights())
    print("✅ Model architecture rebuilt and weights copied!")

    # ✅ Save intermediate HDF5
    print("💾 Saving intermediate HDF5 model...")
    rebuilt_model.save(INTERMEDIATE_MODEL_PATH, save_format="h5")

    # ✅ Reload from HDF5
    print("📥 Reloading intermediate model from HDF5...")
    fixed_model = tf.keras.models.load_model(INTERMEDIATE_MODEL_PATH, compile=False)

    # ✅ Force rebuild to avoid weight mismatch
    print("🔁 Cloning model structure and loading weights...")
    new_model = tf.keras.models.clone_model(fixed_model)
    new_model.set_weights(fixed_model.get_weights())

    # ✅ Optional: check weights match
    print("🔍 Verifying weight match...")
    for w1, w2 in zip(fixed_model.get_weights(), new_model.get_weights()):
        if not (w1 == w2).all():
            print("⚠️ Warning: Weight mismatch detected!")
            break
    else:
        print("✅ Weight match confirmed!")

    # ✅ Save final .keras model
    print("💾 Saving final `.keras` model...")
    new_model.save(NEW_MODEL_PATH, save_format="keras")
    print(f"🎉 Final model saved successfully at: {NEW_MODEL_PATH}")

except Exception as e:
    print(f"❌ Error during conversion: {e}")
