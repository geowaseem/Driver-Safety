# driversafety
This project implements a real-time driver assistance system using computer vision and deep learning techniques. It leverages the ViT (Vision Transformer) model for facial expression recognition and the YOLO-v8 object detection framework for identifying driver behaviors within the car environment.

Features:
- Facial Expression Recognition: Utilizes ViT model to analyze facial expressions and assess the driver's emotional state in real-time.
- Object Detection: Implements YOLO-v8 for identifying various activities such as eating, drinking, seatbelt usage, and signs of drowsiness within the car.
- Real-time Analysis: Provides real-time monitoring and analysis of driver behavior and emotional state for enhanced road safety.
- Audio Feedback: Converts the predicted emotions into speech using gTTS (Google Text-to-Speech) for audio feedback to the driver.

Requirements:
- Python 3.x
- PyTorch
- torchvision
- transformers
- OpenCV
- pygame
- gTTS
- ultralytics

Usage:
1. Install the required libraries mentioned in the requirements.txt.
2. Ensure proper setup of PyTorch, OpenCV, and other dependencies.
3. Place the pre-trained weights file weights.pt in the specified directory.
4. Prepare Dataset 1 and Dataset 2 for training and testing as described below.

Dataset Preparation:
- Dataset 1: Contains 700 images of faces for training and testing the ViT model for facial expression recognition.
- Dataset 2: Consists of over 5000 images used for training both ViT and YOLO-v8 models, providing diverse images for real-time identification systems.


Running the System:
- Execute the ViT code for facial expression recognition and YOLO-v8 code for object detection.
- The ViT model analyzes facial expressions from the camera feed, while YOLO-v8 detects and classifies driver behaviors.
- The system provides real-time audio feedback based on the detected emotions and behaviors.

Acknowledgments:
- The project utilizes the ViT model and YOLO-v8 framework, both of which are open-source libraries.
- Credits to the respective authors and contributors for their valuable contributions.

License:
This project is licensed under the MIT License.
