# 🎥 Deep Fake Video Detection

This project presents an AI-powered Deep Fake Video Detection system using a hybrid deep learning approach combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), specifically GRUs, for detecting manipulated videos with high accuracy.


🚩 Problem Statement

Deepfakes, synthetic manipulated videos created using advanced GANs, pose critical threats to media authenticity, politics, cybersecurity, and public trust. This system aims to automatically detect such fake videos by analyzing spatial features in individual frames and temporal inconsistencies across frames to combat misinformation effectively.


✨ Key Features

🧠 Hybrid CNN-GRU Architecture: Extract spatial features with CNN and model temporal dependencies with GRU for improved detection accuracy.

📊 Frame-level Video Analysis: Processes and classifies individual video frames to detect anomalies.

⚙️ Modular Pipeline: Frame extraction, preprocessing, feature extraction, temporal modeling, classification, and evaluation modules.

📈 Performance Evaluation: Includes accuracy, precision, recall, F1-score, confusion matrix, and ROC curve for comprehensive model assessment.

💻 Real-world Dataset: Utilizes the DeepFake Detection Challenge (DFDC) Kaggle dataset (4GB subset) containing authentic and manipulated videos.

🎯 Transfer Learning: Leverages pretrained CNN backbones (e.g., InceptionV3) to reduce training time and enhance feature extraction.

🛠️ Scalable & Extensible: Designed to scale and support future improvements like transformer models, explainable AI, and real-time deployment.


🏗️ Tech Stack

Programming Language: Python

Deep Learning Frameworks: TensorFlow, Keras

Computer Vision: OpenCV for frame extraction and face detection

ML Utilities: Scikit-learn for metrics and evaluation

Visualization: Matplotlib, Seaborn for plots and interpretations

Development Environment: Jupyter Notebook


📂 Project Structure

Data Preprocessing: Extracts and normalizes video frames to fixed size.

Feature Extraction: Uses pretrained CNN to convert frames to feature vectors.

Temporal Modeling: GRU layers capture sequential dependencies across frames.

Classification: Fully connected sigmoid layer classifies videos as REAL or FAKE.

Evaluation: Metrics and visualization tools ensure robust performance analysis.


🚀 How to Run

1. Clone the repository:

bash

git clone https://github.com/your-username/deepfake-video-detection.git

cd deepfake-video-detection

2. Install dependencies (recommend using a virtual environment):

bash

pip install -r requirements.txt

3. Prepare the DFDC dataset subset and update dataset paths in scripts.

4. Run the Jupyter Notebook or Python scripts for training and testing the model.


📈 Model Performance

Accuracy: ~80% on validation dataset

Precision & Recall: Balanced performance indicating reliable classification

Visual Evaluation: Confusion matrices and ROC curves illustrate model robustness

Generalization: Able to detect various manipulation techniques in diverse videos


🔮 Future Scope

Integrate transformer-based models and 3D CNNs for improved spatio-temporal learning.

Implement explainable AI techniques (e.g., Grad-CAM) for interpretability.

Optimize for low-resource real-time inferencing on edge devices.

Extend detection to multimodal inputs (audio + video).

Build adversarial robust systems to combat evolving deepfake technologies.


🤝 Contribution

Contributions, suggestions, and bug reports are welcomed. Please fork the repo, create feature branches, and submit pull requests.


📜 License

Released under the MIT License.

