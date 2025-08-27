# Face Detection and Data Analysis with OpenCV

This project uses **OpenCV** to detect faces from a webcam, saves cropped face images, and records metadata (time, width, height) into a CSV file.  
Later, the data is analyzed and visualized with **Pandas**, **Matplotlib**, and **Seaborn**.

---

## 🚀 Features
- Real-time face detection using Haar Cascade.
- Saves cropped face images into a `faces/` folder.
- Records metadata (image path, time, width, height) into `faces_metadata.csv`.
- Provides data analysis:
  - Number of detected faces per minute.
  - Distribution of face widths and heights.

---

## 🛠️ Requirements
Make sure you have the following Python packages installed:

```bash
pip install opencv-python pandas matplotlib seaborn
