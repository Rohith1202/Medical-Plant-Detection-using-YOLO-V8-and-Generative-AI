# 🌿 Medical Plant Detection using YOLOv8 and Generative AI

## 🧠 Project Overview

This project focuses on the development of an intelligent and user-friendly web application that can accurately detect **medicinal plants** using deep learning techniques. By leveraging the power of **YOLOv8 (You Only Look Once)** object detection model and **Streamlit**, this application allows users to upload images of plants and receive real-time predictions on the identified medicinal species.

This tool is designed to support **botanical research**, **herbal medicine identification**, and **educational purposes**, especially in rural and tribal areas where access to professional botanical expertise may be limited.

---

## 🎯 Objectives

- Build a lightweight and accurate object detection model capable of recognizing multiple medicinal plants.
- Deploy an interactive web interface using Streamlit for ease of use.
- Enable image upload, display the most confident plant classification, and provide downloadable reports.
- Maintain user-specific logs with timestamped detection data (name, age, and results).

---

## 🛠️ Technology Stack

| Component            | Description                                 |
|----------------------|---------------------------------------------|
| **Model**            | YOLOv8 (Ultralytics) pre-trained + custom   |
| **Framework**        | PyTorch, Streamlit                          |
| **Language**         | Python                                      |
| **Interface**        | Streamlit-based UI                          |
| **Tools & Libraries**| OpenCV, NumPy, Pandas, PIL, os, datetime    |

---

## 🔍 Model Training & Optimization

- The YOLOv8 model was trained using a **custom dataset** consisting of labeled medicinal plant images.
- Training involves:
  - Data preprocessing and augmentation
  - Bounding box annotations in YOLO format
  - Model fine-tuning and validation
- The final model is exported as `best.pt`, ready for inference.

---

## 📷 Streamlit Application Features

- 🌱 Upload an image for medicinal plant detection.
- 🔎 Detect and show **only the most confident** plant name from the image.
- 🧾 Display the result clearly below the image.
- 🧑‍💼 Collect user name and age for logging purposes.
- ⏰ Show a **real-time clock** on the interface.
- 📥 Download detection results and image as a `.txt` log file.
- ✨ Custom UI with background styling for an enhanced experience.

---

## 📁 File Structure

```
├── Medical Plant Detection.ipynb   # Jupyter Notebook for model loading and Streamlit app
├── best.pt                         # Trained YOLOv8 model
├── plant_images/                   # Folder with test or sample plant images
├── requirements.txt                # Required Python packages
├── utils/                          # Helper functions (optional)
└── README.md                       # Project documentation
```

---

## 🧪 Sample Use Case

1. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

2. Upload a plant image.

3. Fill in your name and age in the provided fields.

4. View the detection result and download the output file.

---

## 📥 Sample Output File (TXT)

```
Name: Anusha
Age: 22
Timestamp: 2025-05-29 17:05:23
Detected Plant: Ocimum tenuiflorum (Tulsi)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- YOLOv8 (`ultralytics` package)
- Streamlit
- OpenCV, Pillow, Pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This project utilizes a comprehensive medical plant detection dataset hosted on Roboflow Universe. The dataset contains high-quality annotated images of various medicinal plants, specifically curated for object detection and classification tasks.

**Dataset URL:** [https://universe.roboflow.com/pkm-cj9yb/plant-detection-iqmnt](https://universe.roboflow.com/pkm-cj9yb/plant-detection-iqmnt)

### Dataset Features
- High-resolution images of medicinal plants
- Accurate bounding box annotations
- Multiple export formats (YOLO, COCO, Pascal VOC)
- Pre-processing and augmentation options
- Suitable for training custom plant detection models

### Usage
The dataset is used to train our deep learning models for accurate plant identification and classification, enabling the system to recognize various medicinal plants with high precision.

### Run the Application

```bash
streamlit run Medical_Plant_Detection.py
```

---

## 🎓 Applications

- Medicinal plant identification in remote areas.
- Assisting researchers and herbal practitioners.
- Building botanical datasets for AI.
- Educational tools for plant taxonomy.

---

## Screenshots

### Authentication

![image](https://github.com/user-attachments/assets/f1dcabb1-d861-4b0d-91e9-8403336ac827)

*User login interface for accessing the medical plant detection system*

### Plant Detection

![image](https://github.com/user-attachments/assets/60807322-5b6f-4752-ac35-e856f965ec05)

*Upload image interface for plant identification*

![image](https://github.com/user-attachments/assets/a6f24aee-a9d0-4e68-bc07-074c55f78a25)

*Real-time plant detection using webcam*

### AI Assistant

![image](https://github.com/user-attachments/assets/bb433dc1-ba79-4170-86a1-245497da9d59)

*Interactive AI chatbot for plant-related queries*

### Detection History

![image](https://github.com/user-attachments/assets/fe2dbbd8-5217-4b0a-9378-705e976f9650)

*View previous plant detection results and analysis*

## Additional Features

### User Feedback

![image](https://github.com/user-attachments/assets/d5729a78-6b51-4a2e-a8f3-c8347b5b9c2b)

*User feedback interface for system improvement*

### About Section

![image](https://github.com/user-attachments/assets/fc0c0794-baf6-4d65-ad3f-6abc5bfd4d0f)

*Information about the medical plant detection system*

---
---

## 👩‍💻 About Me

Hi, I’m **Anusha Pantala** — a passionate and driven **final-year B.Tech student** in **Computer Science and Engineering** with a specialization in **Data Science**.

I'm deeply interested in building real-world tech solutions that combine data, intelligence, and intuitive design. My academic journey and hands-on projects reflect a strong foundation in both theory and practical application.

### 👇 My Core Interests
- 🔍 Data Science & Analytics  
- 🤖 Artificial Intelligence & Machine Learning  
- 🌐 Full-Stack Web Development  
- 📊 BI Dashboards & Predictive Modeling  
- 💡 Problem-Solving with Scalable Technologies

I enjoy translating business needs and data insights into impactful software solutions that solve real problems and enhance user experiences.

---

## 🔗 Let’s Connect

📫 **LinkedIn**  
Let’s connect and grow professionally:  
[linkedin.com/in/pantala-anusha](https://www.linkedin.com/in/pantala-anusha/)

🌐 **Portfolio**  
Explore my latest work, skills, and projects here:  
[anusha-pantala.vercel.app](https://anusha-pantala.vercel.app)
