import os
import json
import shutil
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, applications
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

# -------------------- USER CONFIG --------------------
PROCESSED_ROOT = r"D:\abhi\FINAL PROJ\affectnet_processed"  # adjust if necessary
CLASS_NAMES = ['Anger','Contempt','Disgust','Fear','Happy','Neutral','Sad','Surprise']
NUM_CLASSES = len(CLASS_NAMES)

# toggles (recommended)
USE_MIXUP = False      # disabled for emotion datasets
USE_CUTMIX = False
USE_FOCAL_LOSS = True


# optimization / augmentation / sizes
LABEL_SMOOTHING = 0.08   # increased smoothing
MIXUP_ALPHA = 0.2

IMG_SIZE = (224, 224)    # improved resolution for MobileNetV2
BATCH_SIZE = 48

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 50
TOTAL_EPOCHS = HEAD_EPOCHS + FINE_TUNE_EPOCHS

BASE_LR = 1e-4
FINE_TUNE_LR = 3e-5   # lower LR for finer updates

SAVE_DIR = "training_outputs"
TOP_K_MISCLASS = 50
MAX_TSNE_SAMPLES = 2000
RANDOM_SEED = 42
# -----------------------------------------------------

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# hardware
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)
if gpus:
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled.")
    except Exception:
        pass
else:
    tf.keras.mixed_precision.set_global_policy('float32')
    print("No GPU detected (CPU-only).")

AUTOTUNE = tf.data.AUTOTUNE

# ---------------- dataset loading ----------------
print("Loading dataset from:", PROCESSED_ROOT)
train_raw = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(PROCESSED_ROOT, 'train'),
    labels='inferred', label_mode='int',
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED
)
val_raw = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(PROCESSED_ROOT, 'val'),
    labels='inferred', label_mode='int',
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
test_raw = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(PROCESSED_ROOT, 'test'),
    labels='inferred', label_mode='int',
    image_size=IMG_SIZE, batch_size=128, shuffle=False
)

train_ds = make_fixed_batch(train_raw, BATCH_SIZE)
val_ds = make_fixed_batch(val_raw, BATCH_SIZE)
test_all_ds = test_raw

normalization = layers.Rescaling(1./255)

def to_onehot(images, labels):
    return images, tf.one_hot(labels, depth=NUM_CLASSES)

train_ds = train_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(to_onehot, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(to_onehot, num_parallel_calls=AUTOTUNE)
test_all_ds = test_all_ds.map(lambda x,y: (normalization(x), tf.one_hot(y, depth=NUM_CLASSES)), num_parallel_calls=AUTOTUNE)

# ---------------- Keras-safe Random Brightness Layer ----------------
class RandomBrightness(layers.Layer):
    def __init__(self, max_delta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta

    def call(self, inputs, training=None):
        # training will be True during model.fit, False during inference
        if training is None:
            # fallback: assume not training
            return inputs
        def aug():
            # tf.image.random_brightness expects floats in range [0,1] (we already rescale)
            return tf.image.random_brightness(inputs, max_delta=self.max_delta)
        return tf.cond(tf.cast(training, tf.bool), lambda: aug(), lambda: inputs)

# augmentation: keep in-model augmentation but stronger
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.06,0.06),
    layers.RandomContrast(0.30),
    RandomBrightness(max_delta=0.25),
], name="augmentation")

# compute train_count robustly
train_count = 0
for cls in CLASS_NAMES:
    p = os.path.join(PROCESSED_ROOT, 'train', cls)
    if os.path.isdir(p):
        train_count += len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f))])
steps_per_epoch = max(1, train_count // BATCH_SIZE)

# ---------------- landmarks (MediaPipe) ----------------
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    mediapipe_available = True
    print("mediapipe available")
except Exception as e:
    mp = None
    mediapipe_available = False
    print("mediapipe NOT available:", e)

# extracts facial landmarks using MediaPipe, converts them into numbers, saves them, and returns them for model training.
# Each image → 468 landmarks × 3 coordinates = 1404 values
def ensure_landmarks(split_name):
    arr_path = os.path.join(PROCESSED_ROOT, f"{split_name}_landmarks.npy")
    labels_path = os.path.join(PROCESSED_ROOT, f"{split_name}_landmarks_labels.npy")

    filepaths, labels = [], []

    # Collect image paths and labels
    for i, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(PROCESSED_ROOT, split_name, cls)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            fp = os.path.join(folder, fname)
            if os.path.isfile(fp):
                filepaths.append(fp)
                labels.append(i)

    n = len(filepaths)
    if n == 0:
        return np.zeros((0, 468*3), np.float32), np.array([], np.int32)

    # Extract landmarks
    landmarks = []
    if mediapipe_available:
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            for fp in filepaths:
                try:
                    img = np.array(Image.open(fp).convert('RGB'))
                    res = face_mesh.process(img)
                    if res.multi_face_landmarks:
                        lm = np.array([[p.x, p.y, p.z] 
                                       for p in res.multi_face_landmarks[0].landmark], np.float32)
                        landmarks.append(lm.reshape(-1))
                    else:
                        landmarks.append(np.zeros(468*3, np.float32))
                except:
                    landmarks.append(np.zeros(468*3, np.float32))
    else:
        landmarks = [np.zeros(468*3, np.float32)] * n

    lm_arr = np.stack(landmarks)
    np.save(arr_path, lm_arr)
    np.save(labels_path, np.array(labels, np.int32))

    return lm_arr, np.array(labels, np.int32)

train_lms, train_labs_arr = ensure_landmarks('train')
val_lms, val_labs_arr = ensure_landmarks('val')
test_lms, test_labs_arr = ensure_landmarks('test')

def build_filepaths_list(split_name):
    filepaths = []
    labels = []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(PROCESSED_ROOT, split_name, cls)
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                fp = os.path.join(folder, fname)
                if os.path.isfile(fp):
                    filepaths.append(fp)
                    labels.append(cls_idx)
    return filepaths, labels

train_files, _ = build_filepaths_list('train')
val_files, _ = build_filepaths_list('val')
test_files, _ = build_filepaths_list('test')

# ---------- Replace landmark datasets + steps_per_epoch computation ----------
landmark_train_ds = tf.data.Dataset.from_tensor_slices(train_lms).repeat().batch(BATCH_SIZE, drop_remainder=True)
landmark_val_ds = tf.data.Dataset.from_tensor_slices(val_lms).batch(BATCH_SIZE, drop_remainder=True)

num_train_landmark_batches = max(1, train_lms.shape[0] // BATCH_SIZE)
steps_per_epoch = num_train_landmark_batches
# ---------------------------------------------------------------------------

# ---------------- custom Keras-safe layers ----------------
def squeeze_excite_block_4d(inputs, se_ratio=8):
    filters = int(inputs.shape[-1])
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1,1,filters))(se)
    se = layers.Conv2D(max(1, filters // se_ratio), kernel_size=1, activation='relu', padding='same')(se)
    se = layers.Conv2D(filters, kernel_size=1, activation='sigmoid', padding='same')(se)
    return layers.Multiply()([inputs, se])

def residual_dense_block(x, units=256, dropout_rate=0.3):
    shortcut = x
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    if int(shortcut.shape[-1]) != units:
        shortcut = layers.Dense(units, activation=None)(shortcut)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def spatial_channel_attention_layer(inputs):
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(inputs)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(inputs)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    att = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([inputs, att])

# ---------------- build model (unchanged backbone style, better head) ----------------
def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES, landmark_dim=468*3):
    base = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False
    img_input = layers.Input(shape=input_shape, name="image_input")
    x = augmentation(img_input)  # use the augmentation we defined above
    x = base(x, training=False)
    x = spatial_channel_attention_layer(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.45)(x)
    img_feat = layers.Dense(256, activation='relu')(x)
    img_feat = layers.BatchNormalization()(img_feat)

    lm_input = layers.Input(shape=(landmark_dim,), name="landmark_input")
    y = layers.Dense(512, activation='relu')(lm_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    lm_feat = layers.Dropout(0.3)(y)

    fusion = layers.Concatenate(name="fusion")([img_feat, lm_feat])  # ~512
    z = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(fusion)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    # two residual dense blocks (extra capacity but stable)
    z = residual_dense_block(z, units=512, dropout_rate=0.4)
    z = residual_dense_block(z, units=512, dropout_rate=0.3)
    # squeeze-excite
    z_se = layers.Reshape((1,1,int(z.shape[-1])))(z)
    z_se = squeeze_excite_block_4d(z_se, se_ratio=8)
    z = layers.Reshape((int(z.shape[-1]),))(z_se)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name="preds")(z)

    model = models.Model(inputs=[img_input, lm_input], outputs=outputs, name="mobilenetv2_mediapipe_v2")
    return model, base

model, backbone = build_model()
model.summary()

# ---------------- loss / optimizer ----------------
if USE_FOCAL_LOSS:
    gamma = 1.5
    alpha = 0.25
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross = - y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * cross
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))
    loss_fn = focal_loss_fn
else:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=LABEL_SMOOTHING)

opt = optimizers.Adam(learning_rate=BASE_LR)
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

# ---------------- callbacks ----------------
ckpt_head = callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, f"best_head_{timestamp}.keras"),
                                      monitor='val_accuracy', save_best_only=True, verbose=1)
ckpt_ft = callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, f"best_finetune_{timestamp}.keras"),
                                    monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
tb = callbacks.TensorBoard(log_dir=os.path.join(SAVE_DIR, "tb_logs"), histogram_freq=0)

# ---------------- class weights ----------------
def compute_class_weights(root, class_names):
    counts = []
    for i,cls in enumerate(class_names):
        p = os.path.join(root, 'train', cls)
        c = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f))]) if os.path.isdir(p) else 0
        counts.append(c)
    counts = np.array(counts, dtype=np.float32)
    total = counts.sum()
    class_weights = {}
    for i,c in enumerate(counts):
        class_weights[i] = float(total / (NUM_CLASSES * c)) if c>0 else 1.0
    print("Class counts:", dict(zip(class_names, counts.tolist())))
    print("Class weights vector:", [round(class_weights[i],6) for i in range(NUM_CLASSES)])
    return class_weights

class_weights = compute_class_weights(PROCESSED_ROOT, CLASS_NAMES)

# ---------------- pair image batches with landmark batches ----------------
train_ds_zipped = tf.data.Dataset.zip((train_ds, landmark_train_ds))
def _map_fn(batch, lms):
    images, labels = batch
    return (images, lms), labels
train_ds_final = train_ds_zipped.map(lambda batch, lms: _map_fn(batch, lms), num_parallel_calls=AUTOTUNE)
train_ds_final = train_ds_final.prefetch(AUTOTUNE)

val_ds_zipped = tf.data.Dataset.zip((val_ds, landmark_val_ds))
val_ds_final = val_ds_zipped.map(lambda batch, lms: _map_fn(batch, lms), num_parallel_calls=AUTOTUNE)
val_ds_final = val_ds_final.prefetch(AUTOTUNE)

# ---------------- Stage 1: train head (backbone frozen) ----------------
print("Stage 1 — training head (backbone frozen)")
backbone.trainable = False
history_head = model.fit(
    train_ds_final,
    validation_data=val_ds_final,
    epochs=HEAD_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[ckpt_head, reduce_lr, tb],
    class_weight=class_weights,
    verbose=1
)

# ---------------- Stage 2: fine-tune ----------------
print("Stage 2 — unfreezing backbone and fine-tuning")
# Unfreeze backbone but keep first N layers frozen (less aggressive freeze)
for layer in backbone.layers[:20]:
    layer.trainable = False
for layer in backbone.layers[20:]:
    layer.trainable = True

# cosine decay schedule for fine-tune optimizer
decay_steps = max(1, steps_per_epoch * FINE_TUNE_EPOCHS)
cosine = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=FINE_TUNE_LR,
    decay_steps=decay_steps
)
opt2 = tf.keras.optimizers.Adam(learning_rate=cosine)
model.compile(optimizer=opt2, loss=loss_fn, metrics=['accuracy'])

initial_epoch = history_head.epoch[-1] + 1 if history_head.epoch else 0
history_ft = model.fit(
    train_ds_final,
    validation_data=val_ds_final,
    epochs=TOTAL_EPOCHS,
    initial_epoch=initial_epoch,
    steps_per_epoch=steps_per_epoch,
    callbacks=[ckpt_ft, reduce_lr, early_stopping, tb],
    class_weight=class_weights,
    verbose=1
)

# ---------------- combine history, save ----------------
history = {}
for k,v in history_head.history.items(): history.setdefault(k,[]).extend(v)
for k,v in history_ft.history.items(): history.setdefault(k,[]).extend(v)
with open(os.path.join(SAVE_DIR, f"history_{timestamp}.json"), "w") as f:
    json.dump(history, f, indent=2)

best_val_acc = 0.0
for key in ('val_accuracy','val_categorical_accuracy'):
    if key in history:
        best_val_acc = max(best_val_acc, max(history[key]) if len(history[key])>0 else 0.0)
best_val_acc = float(best_val_acc)

final_model_path = os.path.join(SAVE_DIR, f"final_model_{timestamp}_valacc{best_val_acc:.4f}.keras")
model.save(final_model_path)
print("Saved final model to:", final_model_path)

# ---------------- Evaluation & visualizations ----------------
print("Evaluating on test set...")
test_batch_size = 128
test_lm_ds = tf.data.Dataset.from_tensor_slices(test_lms).batch(test_batch_size)
test_all_images = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(PROCESSED_ROOT, 'test'),
    labels='inferred', label_mode='int',
    image_size=IMG_SIZE, batch_size=test_batch_size, shuffle=False
)
test_all_images = test_all_images.map(lambda x,y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
test_zipped = tf.data.Dataset.zip((test_all_images, test_lm_ds))
def _map_test(batch, lms):
    images, labels = batch
    return (images, lms), tf.one_hot(labels, depth=NUM_CLASSES)
test_all = test_zipped.map(_map_test, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

y_true = []
y_prob = []
for (imgs, lms), labels_onehot in test_all:
    preds = model.predict([imgs, lms], verbose=0)
    y_prob.append(preds)
    y_true.append(np.argmax(labels_onehot.numpy(), axis=1))

if len(y_prob) == 0:
    print("No test batches found. Exiting evaluation.")
    exit(0)

y_prob = np.vstack(y_prob)
y_true = np.concatenate(y_true)
y_pred = np.argmax(y_prob, axis=1)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
acc = accuracy_score(y_true, y_pred)
prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f"Accuracy: {acc:.4f}  Macro Precision: {prec_macro:.4f}  Macro Recall: {rec_macro:.4f}  Macro F1: {f1_macro:.4f}")

metrics_summary = {"accuracy": float(acc), "precision_macro": float(prec_macro),
                   "recall_macro": float(rec_macro), "f1_macro": float(f1_macro),
                   "best_val_acc": best_val_acc}
with open(os.path.join(SAVE_DIR, f"metrics_summary_{timestamp}.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
cm_path = os.path.join(SAVE_DIR, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, bbox_inches='tight'); plt.close()

cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
plt.figure(figsize=(9,7))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Normalized Confusion Matrix')
cmn_path = os.path.join(SAVE_DIR, f"confusion_matrix_normalized_{timestamp}.png")
plt.savefig(cmn_path, bbox_inches='tight'); plt.close()
print("Saved confusion matrices to:", cm_path, cmn_path)

# ROC / PR / support / learning curves / misclassified / t-SNE (same plan as before)
y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
plt.figure(figsize=(10,8))
roc_auc = {}
for i in range(NUM_CLASSES):
    try:
        fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_prob[:,i])
        roc_auc[i] = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:0.3f})")
    except Exception:
        roc_auc[i] = float('nan')
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title('ROC Curves')
roc_path = os.path.join(SAVE_DIR, f"roc_curves_{timestamp}.png")
plt.savefig(roc_path, bbox_inches='tight'); plt.close()
for i in range(NUM_CLASSES):
    print(f"AUC - {CLASS_NAMES[i]}: {roc_auc[i]:.4f}")

plt.figure(figsize=(10,8))
pr_auc = {}
for i in range(NUM_CLASSES):
    try:
        precision, recall, _ = precision_recall_curve(y_true_bin[:,i], y_prob[:,i])
        ap = average_precision_score(y_true_bin[:,i], y_prob[:,i])
        pr_auc[i] = ap
        plt.plot(recall, precision, label=f"{CLASS_NAMES[i]} (AP={ap:0.3f})")
    except Exception:
        pr_auc[i] = float('nan')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curves')
plt.legend()
pr_path = os.path.join(SAVE_DIR, f"pr_curves_{timestamp}.png")
plt.savefig(pr_path, bbox_inches='tight'); plt.close()
for i in range(NUM_CLASSES):
    print(f"PR-AUC - {CLASS_NAMES[i]}: {pr_auc[i]:.4f}")

support_counts = [np.sum(y_true == i) for i in range(NUM_CLASSES)]
plt.figure(figsize=(10,4))
sns.barplot(x=CLASS_NAMES, y=support_counts)
plt.ylabel('Support (test samples)'); plt.title('Per-class support on test set')
plt.xticks(rotation=45)
supp_path = os.path.join(SAVE_DIR, f"class_support_{timestamp}.png")
plt.savefig(supp_path, bbox_inches='tight'); plt.close()
print("Saved class support chart to:", supp_path)

train_acc = history.get('accuracy', []) or history.get('categorical_accuracy', [])
val_acc = history.get('val_accuracy', []) or history.get('val_categorical_accuracy', [])
train_loss = history.get('loss', [])
val_loss = history.get('val_loss', [])
epochs_range = range(1, len(train_acc) + 1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, train_acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs_range, train_loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
lc_path = os.path.join(SAVE_DIR, f"learning_curves_{timestamp}.png")
plt.savefig(lc_path, bbox_inches='tight'); plt.close()

print(f"Saving top-{TOP_K_MISCLASS} high-confidence misclassified test images to disk...")
mis_dir = os.path.join(SAVE_DIR, f"misclassified_{timestamp}")
if os.path.exists(mis_dir):
    shutil.rmtree(mis_dir)
os.makedirs(mis_dir, exist_ok=True)

confidences = np.max(y_prob, axis=1)
wrong_idxs = np.where(y_pred != y_true)[0]
if len(wrong_idxs) == 0:
    print("No misclassifications found on test set.")
else:
    wrong_conf = confidences[wrong_idxs]
    sorted_idx = wrong_idxs[np.argsort(-wrong_conf)]
    to_save = sorted_idx[:TOP_K_MISCLASS]
    filepaths = np.array(test_files)
    for rank, idx in enumerate(to_save):
        try:
            fp = filepaths[idx]
            true_label = CLASS_NAMES[int(y_true[idx])]
            pred_label = CLASS_NAMES[int(y_pred[idx])]
            conf = float(confidences[idx])
            im = Image.open(fp).convert('RGB').resize((400,400))
            draw = ImageDraw.Draw(im)
            text = f"true: {true_label} | pred: {pred_label} | conf: {conf:.3f}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            draw.rectangle([(0,0),(400,40)], fill=(0,0,0,127))
            draw.text((6,6), text, fill=(255,255,255), font=font)
            outp = os.path.join(mis_dir, f"{rank:03d}_idx{idx}_true_{true_label}_pred_{pred_label}_conf{conf:.3f}.png")
            im.save(outp)
        except Exception as e:
            print("Error saving misclassified image idx", idx, e)
    print("Saved misclassified images to:", mis_dir)

# t-SNE of penultimate features (subset)
print("Computing t-SNE embeddings of penultimate features (subset)...")
filepaths_arr = np.array(test_files)
n_total = len(filepaths_arr)
if n_total == 0:
    print("No test images found for t-SNE.")
else:
    sel_idx = np.linspace(0, n_total-1, min(MAX_TSNE_SAMPLES, n_total), dtype=int)
    imgs = []
    labs = []
    for i in sel_idx:
        try:
            im = Image.open(filepaths_arr[i]).convert('RGB').resize(IMG_SIZE)
            arr = np.array(im) / 255.0
            imgs.append(arr)
            labs.append(test_labs_arr[i])
        except:
            pass
    if len(imgs)==0:
        print("No images loaded for t-SNE.")
    else:
        imgs = np.stack(imgs, axis=0).astype(np.float32)
        labs = np.array(labs)
        # penultimate layer extraction
        penult_layer = None
        for i,lay in enumerate(model.layers[::-1]):
            if isinstance(lay, tf.keras.layers.Dense) and lay.name == 'preds':
                penult_idx = len(model.layers) - (i+1) - 1
                penult_layer = model.layers[penult_idx].output
                break
        if penult_layer is None:
            penult_layer = model.layers[-2].output
        feat_extractor = models.Model(inputs=model.input, outputs=penult_layer)
        lm_subset = test_lms[sel_idx][:len(imgs)]
        feats = feat_extractor.predict([imgs, lm_subset], batch_size=64, verbose=0)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=RANDOM_SEED)
        embeddings = tsne.fit_transform(feats)
        plt.figure(figsize=(10,8))
        palette = sns.color_palette("hsv", NUM_CLASSES)
        for i,cname in enumerate(CLASS_NAMES):
            idxs = np.where(labs==i)[0]
            if len(idxs)>0:
                plt.scatter(embeddings[idxs,0], embeddings[idxs,1], label=cname, s=8)
        plt.legend(markerscale=2, bbox_to_anchor=(1.05,1), loc='upper left')
        plt.title("t-SNE (penultimate features) — subset of test samples")
        tsne_path = os.path.join(SAVE_DIR, f"tsne_features_{timestamp}.png")
        plt.savefig(tsne_path, bbox_inches='tight'); plt.close()
        print("Saved t-SNE plot to:", tsne_path)

print("ALL DONE. Outputs in:", SAVE_DIR)
