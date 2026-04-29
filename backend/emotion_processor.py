import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
from pathlib import Path
from collections import deque

tf.config.run_functions_eagerly(True)

from tensorflow.keras import layers, models, applications
from keras import config
config.enable_unsafe_deserialization()

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "training_outputs" / "best_finetune_20260125_224238.keras"

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)

EMOTION_LABELS = [
    'Anger', 'Contempt', 'Disgust', 'Fear',
    'Happy', 'Neutral', 'Sad', 'Surprise'
]

NUM_CLASSES = len(EMOTION_LABELS)

EMOTION_MODEL = None
BACKBONE_LAYER = None
FACE_DETECTION = None
FACE_MESH = None

PRED_HISTORY = deque(maxlen=1)

# ---------------- MODEL ----------------
def spatial_channel_attention_layer(inputs):
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(inputs)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(inputs)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    att = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([inputs, att])

def build_model(input_shape=(224, 224, 3), landmark_dim=468 * 3):
    base = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    img_input = layers.Input(shape=input_shape, name="image_input")
    x = base(img_input, training=False)
    x = spatial_channel_attention_layer(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)

    lm_input = layers.Input(shape=(landmark_dim,), name="landmark_input")
    y = layers.Dense(256, activation='relu')(lm_input)

    fusion = layers.Concatenate()([x, y])
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(fusion)

    return models.Model([img_input, lm_input], outputs)

# ---------------- LOAD MODEL ----------------
def load_models():
    global EMOTION_MODEL, BACKBONE_LAYER, FACE_DETECTION, FACE_MESH

    print("⏳ Loading model...")
    print("Model path:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    EMOTION_MODEL = build_model()
    EMOTION_MODEL.load_weights(str(MODEL_PATH), skip_mismatch=True)

    for layer in reversed(EMOTION_MODEL.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            BACKBONE_LAYER = layer
            print("✅ GradCAM Layer:", layer.name)
            break

    FACE_DETECTION = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1
    )

    print("✅ Model + MediaPipe ready")

# ---------------- GRAD CAM ----------------
def generate_gradcam(img_input, lm_input, class_index):
    grad_model = tf.keras.models.Model(
        inputs=EMOTION_MODEL.inputs,
        outputs=[BACKBONE_LAYER.output, EMOTION_MODEL.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_input, lm_input], training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# ---------------- HELPER ----------------
def get_attention_and_reason(emotion, confidence):
    weights = {
        "Happy": 1.0, 
        "Neutral": 0.8,  
        "Surprise": 0.7, 
        "Fear": 0.4, 
        "Sad": 0.3, 
        "Contempt": 0.3, 
        "Anger": 0.2, 
        "Disgust": 0.2
    }

    base_score = weights.get(emotion, 0.5)

    attentiveness_score = base_score * (0.6 + (0.4 * confidence))

    if attentiveness_score >= 0.70:
        reason = "Engaged & Focused"
    elif attentiveness_score >= 0.40:
        reason = "Slightly Distracted"
    else:
        reason = "Low Focus / Wandering"

    return float(attentiveness_score), reason

# ---------------- FRAME PROCESS ----------------
def process_frame(frame: np.ndarray, student_id="student123"):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = FACE_DETECTION.process(rgb)

    if not results.detections:
        return {
            "emotion": "No Face",
            "probabilities": [],
            "attentiveness_score": 0.0,
            "reason": "no face detected",
            "landmarks": [0.0] * (468 * 3)
        }

    det = results.detections[0]
    box = det.location_data.relative_bounding_box
    h, w, _ = frame.shape

    # Calculate base coordinates
    base_x = int(box.xmin * w)
    base_y = int(box.ymin * h)
    base_w = int(box.width * w)
    base_h = int(box.height * h)

    # ADD 20% PADDING TO FIX THE "SAD" DETECTION ISSUE
    pad_x = int(base_w * 0.20)
    pad_y = int(base_h * 0.20)

    x = max(0, base_x - pad_x)
    y = max(0, base_y - pad_y)
    x2 = min(w, base_x + base_w + pad_x)
    y2 = min(h, base_y + base_h + pad_y)

    face = frame[y:y2, x:x2]

    if face.size == 0:
        return {
            "emotion": "No Face",
            "probabilities": [],
            "attentiveness_score": 0.0,
            "reason": "invalid face crop",
            "landmarks": [0.0] * (468 * 3)
        }

    face_resized = cv2.resize(face, IMG_SIZE)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    img_input = face_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    lm_results = FACE_MESH.process(face_rgb)
    if lm_results.multi_face_landmarks:
        lm = np.array(
            [[p.x, p.y, p.z] for p in lm_results.multi_face_landmarks[0].landmark],
            dtype=np.float32
        )
        lm_flat = lm.reshape(-1)
    else:
        lm_flat = np.zeros((468 * 3,), dtype=np.float32)

    lm_input = np.expand_dims(lm_flat, axis=0)

    preds = EMOTION_MODEL.predict([img_input, lm_input], verbose=0)[0]
    emotion_idx = int(np.argmax(preds))
    current_conf = float(preds[emotion_idx])
    
    # Debug print to terminal so you can monitor live confidence scores
    print(f"Probabilities -> {dict(zip(EMOTION_LABELS, [round(float(p), 2) for p in preds]))}")

    if current_conf > 0.60:
        final_idx = emotion_idx
        PRED_HISTORY.clear()
        PRED_HISTORY.append(final_idx)
    else:
        PRED_HISTORY.append(emotion_idx)
        final_idx = max(set(PRED_HISTORY), key=PRED_HISTORY.count)

    predicted_emotion = EMOTION_LABELS[final_idx]
    probabilities = preds
    confidence = float(preds[final_idx])

    attentiveness_score, reason = get_attention_and_reason(predicted_emotion, confidence)

    try:
        heatmap = generate_gradcam(img_input, lm_input, final_idx)
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face_resized, 0.6, heatmap_color, 0.4, 0)

        combined = np.hstack([face_resized, heatmap_color, overlay])

        explainability_dir = BASE_DIR / "explainability"
        explainability_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(explainability_dir / f"{student_id}_gradcam.png"), combined)
    except Exception as e:
        print(f"GradCAM generation warning: {e}")

    return {
        "emotion": predicted_emotion,
        "probabilities": probabilities.tolist(),
        "attentiveness_score": float(attentiveness_score),
        "reason": reason,
        "landmarks": lm_input.flatten().tolist()
    }