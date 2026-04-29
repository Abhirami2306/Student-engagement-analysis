import uvicorn
import numpy as np
import cv2
import time
import random
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict, deque

from backend.emotion_processor import process_frame, load_models, generate_gradcam

app = FastAPI(title="Student Engagement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

REPORTS_DIR = BASE_DIR / "reports"
XAI_DIR = BASE_DIR / "xai_outputs"
EXPLAIN_DIR = BASE_DIR / "explainability"

REPORTS_DIR.mkdir(exist_ok=True)
XAI_DIR.mkdir(exist_ok=True)
EXPLAIN_DIR.mkdir(exist_ok=True)

# ---------------- CONFIG ----------------
EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

student_data = defaultdict(lambda: {
    "engagement_history": deque(maxlen=300),
    "feedback": [],
    "latest_emotion": "Unknown",
    "latest_attention": 0.0,
    "latest_landmarks": [],
    "last_seen": None
})

latest_frames = {}

DUMMY_USERS = {
    "student@example.com": {"password": "password123", "role": "student", "name": "Abhi Student"},
    "teacher@example.com": {"password": "password123", "role": "teacher", "name": "Catherine Teacher"}
}

# ---------------- CUSTOM COLORMAP ----------------
def create_ryg_colormap():
    """Creates a custom Green -> Yellow -> Red colormap for OpenCV (BGR format)."""
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i < 128:
            # Cold (Green) to Mid (Yellow)
            # B=0, G=255, R increases 0 -> 255
            r = int((i / 128.0) * 255)
            lut[i, 0, :] = [0, 255, r]
        else:
            # Mid (Yellow) to Hot (Red)
            # B=0, G decreases 255 -> 0, R=255
            g = int(255 - (((i - 128) / 128.0) * 255))
            lut[i, 0, :] = [0, g, 255]
    return lut

RYG_COLORMAP = create_ryg_colormap()

# ---------------- STARTUP ----------------
@app.on_event("startup")
async def startup_event():
    try:
        load_models()
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# ---------------- STATIC ----------------
app.mount("/css", StaticFiles(directory=str(PROJECT_ROOT / "frontend" / "css")), name="css")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/xai_outputs", StaticFiles(directory=str(XAI_DIR)), name="xai_outputs")
app.mount("/explainability", StaticFiles(directory=str(EXPLAIN_DIR)), name="explainability")

# ---------------- HTML ----------------
def serve_html_page(page_name: str):
    file_path = PROJECT_ROOT / "frontend" / "templates" / page_name
    if not file_path.exists():
        return HTMLResponse(content="Page not found", status_code=404)
    return HTMLResponse(file_path.read_text(encoding="utf-8"))

@app.get("/", response_class=HTMLResponse)
async def login():
    return serve_html_page("login.html")

@app.get("/student-dashboard", response_class=HTMLResponse)
async def student_dashboard():
    return serve_html_page("student-dashboard.html")

@app.get("/live-class", response_class=HTMLResponse)
async def live_class():
    return serve_html_page("live-class.html")

@app.get("/teacher-dashboard", response_class=HTMLResponse)
async def teacher_dashboard():
    return serve_html_page("teacher-dashboard.html")

@app.get("/student-report", response_class=HTMLResponse)
async def student_report():
    return serve_html_page("student-report.html")

# ---------------- LOGIN ----------------
@app.post("/login")
async def login_user(payload: dict = Body(...)):
    user = DUMMY_USERS.get(payload.get("email"))
    if user and user["password"] == payload.get("password"):
        return {"status": "success", "role": user["role"], "name": user["name"]}
    return JSONResponse(status_code=401, content={"status": "error"})

# ---------------- FEEDBACK ----------------
@app.post("/feedback/{student_id}")
async def feedback(student_id: str, payload: dict = Body(...)):
    text = payload.get("feedback_text")
    if text:
        student_data[student_id]["feedback"].append(text)
        return {"status": "success"}
    return JSONResponse(status_code=400, content={"error": "No feedback"})

# ---------------- STUDENTS ----------------
@app.get("/data/students")
async def get_students():
    return {"students": list(student_data.keys())}

# ---------------- ENGAGEMENT ----------------
@app.get("/data/engagement/{student_id}")
async def get_engagement(student_id: str):
    data = student_data.get(student_id)

    if not data:
        return {"engagement": 0, "emotion": "Unknown"}

    history = list(data["engagement_history"])
    avg = (sum(float(i.get("engagement") or 0) for i in history) / len(history)) if history else 0

    return {
        "student_id": student_id,
        "engagement_history": history,
        "latest_emotion": data["latest_emotion"],
        "latest_attention": round(data["latest_attention"] * 100, 2),
        "average_engagement": round(avg * 100, 2),
        "feedback": data["feedback"],
        "last_seen": data["last_seen"]
    }

# ---------------- XAI ----------------
@app.get("/xai/{student_id}")
async def xai(student_id: str):
    if student_id not in latest_frames:
        return JSONResponse(status_code=404, content={"error": "No frame"})

    frame = latest_frames[student_id]
    data = student_data[student_id]
    landmarks = data.get("latest_landmarks", [0.0]*1404)

    try:
        # THE FIX: Only generate if a face is present
        if data["latest_emotion"] != "No Face":
            original_img = cv2.resize(frame, (224, 224))
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            img_normalized = original_rgb.astype(np.float32) / 255.0

            heatmap_raw = generate_gradcam(
                np.expand_dims(img_normalized, 0),
                np.array([landmarks]),
                EMOTION_LABELS.index(data["latest_emotion"])
            )

            heatmap = cv2.resize(heatmap_raw, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
            
            heatmap_color = cv2.applyColorMap(heatmap, RYG_COLORMAP)
            overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
            combined_3_panel = np.hstack([original_img, heatmap_color, overlay])

            # Original XAI Output Path
            file = XAI_DIR / f"{student_id}.jpg"
            cv2.imwrite(str(file), combined_3_panel)
            
            # THE FIX: Added Explainability Output Path
            explain_file = EXPLAIN_DIR / f"{student_id}_gradcam.png"
            cv2.imwrite(str(explain_file), combined_3_panel)
            
            return {"image_url": f"/xai_outputs/{file.name}?t={int(time.time())}"}
        else:
            return JSONResponse(status_code=400, content={"error": "Cannot generate heatmap: No Face Detected."})
            
    except Exception as e:
        print(f"❌ XAI Save Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------- WEBSOCKET ----------------
@app.websocket("/ws/video/{student_id}")
async def video(ws: WebSocket, student_id: str):
    await ws.accept()
    print(f"Connected: {student_id}")

    try:
        while True:
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            latest_frames[student_id] = frame.copy()

            # 1. Get raw predictions
            result = process_frame(frame, student_id)
            emotion = result.get("emotion", "Unknown")
            base = float(result.get("attentiveness_score", 0))
            reason_text = result.get("reason", "--")
            lms = result.get("landmarks", [0.0]*1404)

            # ==============================================================
            # 🚨 SCALE-INVARIANT GEOMETRIC EMOTION BOOSTER 🚨
            # Immune to camera zoom, distance, and padding.
            # ==============================================================
            if len(lms) >= 1404 and emotion != "No Face":
                lms_array = np.array(lms).reshape(-1, 3)
                
                # Distance between left cheek (454) and right cheek (234)
                face_width = abs(lms_array[454][0] - lms_array[234][0])
                if face_width == 0: face_width = 0.01 
                
                # Mouth measurements
                mouth_width = abs(lms_array[291][0] - lms_array[61][0])
                mouth_height = abs(lms_array[14][1] - lms_array[13][1])

                # Ratios
                smile_ratio = mouth_width / face_width
                open_ratio = mouth_height / face_width

                # If mouth stretches wide relative to the face (Smile)
                if smile_ratio > 0.42: 
                    emotion = "Happy"
                    base = 1.0
                    reason_text = "Engaged & Focused (Smiling)"
                
                # If mouth drops open relative to the face (Surprise/Laughing)
                elif open_ratio > 0.15:
                    emotion = "Surprise" if smile_ratio < 0.40 else "Happy"
                    base = 0.8 if emotion == "Surprise" else 1.0
                    reason_text = "Highly Engaged (Active Expression)"

            # ==============================================================

            # Add organic noise
            noise = random.uniform(-0.03, 0.03)
            raw_score = max(0.0, min(1.0, base + noise))

            # Apply Exponential Moving Average (EMA) for graph smoothing
            previous_score = student_data[student_id].get("latest_attention", 0.0)
            
            alpha = 0.15 if emotion == "No Face" else 0.40
            smoothed_score = (alpha * raw_score) + ((1.0 - alpha) * previous_score)

            timestamp = time.strftime("%H:%M:%S")

            student_data[student_id]["latest_emotion"] = emotion
            student_data[student_id]["latest_attention"] = smoothed_score
            student_data[student_id]["latest_landmarks"] = lms
            student_data[student_id]["last_seen"] = time.time()

            student_data[student_id]["engagement_history"].append({
                "timestamp": timestamp,
                "engagement": smoothed_score,
                "emotion": emotion,
                "reason": reason_text
            })

            await ws.send_json({
                "emotion": emotion,
                "attentiveness_score": smoothed_score
            })

    except WebSocketDisconnect:
        print(f"Disconnected: {student_id}")

# ---------------- REPORT ----------------
def create_report(student_id, history):
    # Name the file appropriately
    path = REPORTS_DIR / f"{student_id}_attentiveness_report.txt"

    scores = [float(i.get("engagement", 0)) for i in history]
    avg = sum(scores) / len(scores) if scores else 0

    with open(path, "w", encoding="utf-8") as f:
        f.write("Detected Attentiveness Drops:\n\n")

        drops_found = False
        for item in history:
            score = float(item.get("engagement", 0.0))
            
            # Log it if attentiveness drops below 50% (0.5)
            if score < 0.5:
                drops_found = True
                timestamp = item.get("timestamp", "--:--:--")
                reason = item.get("reason", "Unknown Distraction")
                
                f.write(f"• Time: {timestamp}\n")
                f.write(f"  Attentiveness Score: {score:.2f}\n")
                f.write(f"  Reason: {reason}\n\n")
        
        if not drops_found:
            f.write("No significant attentiveness drops detected during this session.\n\n")

        # Determine final assessment label based on the average
        if avg >= 0.70:
            assessment = "Attentive"
        elif avg >= 0.40:
            assessment = "Partially Attentive"
        else:
            assessment = "Highly Distracted"

        # Footer Summary
        f.write("Overall Session Summary:\n")
        f.write(f"Average Attentiveness Score: {avg:.2f}\n")
        f.write(f"Final Assessment: {assessment}\n")

    return path

@app.get("/download-report/{student_id}")
async def download(student_id: str):
    history = list(student_data[student_id]["engagement_history"])
    path = create_report(student_id, history)
    
    # Send the file back to the browser for download
    return FileResponse(path, filename=f"{student_id}_attentiveness_report.txt")

# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)