import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.mixed_precision import Policy  # ✅ Correct import
from keras.utils import custom_object_scope
from tensorflow.keras.models import Model

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

    print("🔄 Rebuilding the model architecture...")

    # ✅ Recreate the model using the same architecture
    inputs = old_model.input  # Get input layer
    outputs = old_model.outputs  # Get output layer
    new_model = Model(inputs=inputs, outputs=outputs, name="rebuilt_model")

    # ✅ Copy weights to the new model
    new_model.set_weights(old_model.get_weights())
    print("✅ Model architecture rebuilt successfully!")

    # ✅ Save as HDF5 format first (fix compatibility)
    print("💾 Saving intermediate HDF5 model...")
    new_model.save(INTERMEDIATE_MODEL_PATH, save_format="h5")

    # ✅ Reload from HDF5 (ensures proper format)
    print("🔄 Reloading from HDF5 model...")
    fixed_model = tf.keras.models.load_model(INTERMEDIATE_MODEL_PATH, compile=False)

    # ✅ Save final `.keras` model
    print("💾 Saving final .keras model...")
    fixed_model.save(NEW_MODEL_PATH, save_format="keras")

    print("✅ Model converted and rebuilt successfully!")

except Exception as e:
    print(f"❌ Error during conversion: {e}")
