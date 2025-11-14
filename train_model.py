import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# 🚀 Configuration
# -----------------------------
BASE_DIR = "dataset_ready"  # your folder
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "test")  # using test data as validation
TEST_DIR = os.path.join(BASE_DIR, "test")

EPOCHS = 10  # reduced for faster training
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
LIMIT_IMAGES = 15000  # limits per generator for speed boost

print("🚀 Starting optimized training (Fast Mode - 10 Epochs)")
print(f"📁 Training data: {TRAIN_DIR}")
print(f"📁 Validation data: {VAL_DIR}")

# -----------------------------
# 🧩 Data Preparation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# -----------------------------
# ⚡ Limit the dataset size for faster training
# -----------------------------
def limited_flow_from_directory(datagen, directory, subset_name, limit):
    gen = datagen.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    gen.samples = min(gen.samples, limit)
    print(f"✅ Using {gen.samples} samples for {subset_name}")
    return gen

train_gen = limited_flow_from_directory(train_datagen, TRAIN_DIR, "training", LIMIT_IMAGES)
val_gen = limited_flow_from_directory(val_datagen, VAL_DIR, "validation", int(LIMIT_IMAGES * 0.3))

# -----------------------------
# 🧠 Model Definition (Simple + Fast CNN)
# -----------------------------
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# -----------------------------
# 💾 Checkpoint & Early Stopping
# -----------------------------
checkpoint = ModelCheckpoint(
    "mask_detector_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
)
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# -----------------------------
# 🚀 Training
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    verbose=1,
)

print("✅ Training complete! Model saved as 'mask_detector_model.keras'")

# -----------------------------
# 📊 Evaluate Model on Test Set
# -----------------------------
test_gen = val_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
loss, acc = model.evaluate(test_gen)
print(f"🎯 Final Test Accuracy: {acc * 100:.2f}%")




# cd FACE_MASK_DETECTOR
# .\venv\Scripts\Activate.ps1n
# python train_model.py
