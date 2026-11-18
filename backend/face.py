import cv2
import numpy as np
import json

from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis

#1 Folders
KNOWN_FACES = Path("known-faces")
LOG_FOLDER = Path("attendance_logs")
LOG_FILE = LOG_FOLDER / "recognition.json"

KNOWN_FACES.mkdir(exist_ok=True)
LOG_FOLDER.mkdir(exist_ok=True)

# Step 2: Load face recognition model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))



# Step 3: Load registered employee faces
known_employees = {}

for image_path in KNOWN_FACES.iterdir():

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    faces = app.get(image)

    if len(faces) == 1:
        known_employees[image_path.stem] = (
            faces[0].normed_embedding
        )

print("Registered employees:", list(known_employees.keys()))


# Step 4: Save recognition details as JSON
def save_recognition(name, similarity):

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    record = {
        "employee_name": name,
        "status": "Recognized",
        "similarity": round(float(similarity), 3),
        "timestamp": timestamp
    }

    records = []

    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as file:
            records = json.load(file)

    records.append(record)

    with open(LOG_FILE, "w") as file:
        json.dump(records, file, indent=4)

    print(json.dumps(record, indent=4))


# Step 5: Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot access webcam.")

# Avoid logging the same person repeatedly
logged_employees = set()

print("Webcam started. Press Q to exit.")

try:
    while True:
        success, frame = camera.read()
        if not success:
            break
        faces = app.get(frame)
        for face in faces:
            best_name = "Unknown"
            best_similarity = -1
            live_embedding = face.normed_embedding

            # Compare with registered employees
            for name, reference_embedding in known_employees.items():

                similarity = np.dot(
                    reference_embedding,
                    live_embedding
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name

            # Recognition threshold
            if best_similarity >= 0.5:
                label = f"{best_name} - Recognized"
                color = (0, 255, 0)

                # Save only once per employee per session
                if best_name not in logged_employees:
                    save_recognition(
                        best_name,
                        best_similarity
                    )
                    logged_employees.add(best_name)

            else:
                label = "Unknown Person"
                color = (0, 0, 255)

            # Draw rectangle and name
            left, top, right, bottom = map(
                int, face.bbox
            )

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("Employee Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
