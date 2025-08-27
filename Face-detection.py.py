import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Create folder for saving face images if it doesn't exist
if not os.path.exists("faces"):
    os.makedirs("faces")

# CSV file to store metadata
csv_file = "faces_metadata.csv"
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["Image", "Time", "Width", "Height"]).to_csv(csv_file, index=False)

# Load Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale (better for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face region
        face = frame[y:y+h, x:x+w]
        
        # Save face image
        face_filename = f"faces/face_{count}.jpg"
        cv2.imwrite(face_filename, face)
        
        # Save metadata to CSV
        new_data = pd.DataFrame([[face_filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), w, h]],
                                columns=["Image", "Time", "Width", "Height"])
        new_data.to_csv(csv_file, mode='a', header=False, index=False)
        
        count += 1
    
    # Show webcam feed with detection
    cv2.imshow("Face Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Face images saved and metadata recorded!")

# ---------------- Data Analysis ---------------- #

# Load metadata
df = pd.read_csv(csv_file)

# Show data summary
print("\n📊 Data Summary:")
print(df.describe())

# Convert Time column to datetime
df["Time"] = pd.to_datetime(df["Time"])

# Plot number of detected faces per minute
plt.figure(figsize=(10, 5))
df.set_index("Time").resample("1min").count()["Image"].plot()
plt.title("Number of Faces Detected per Minute")
plt.xlabel("Time")
plt.ylabel("Count")
plt.show()

# Plot histogram of face widths and heights
plt.figure(figsize=(10, 5))
sns.histplot(df["Width"], kde=True, color="blue", label="Width")
sns.histplot(df["Height"], kde=True, color="red", label="Height")
plt.legend()
plt.title("Distribution of Face Widths and Heights")
plt.show()
