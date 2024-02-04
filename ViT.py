from transformers import ViTForImageClassification
from torchvision import transforms
import torch
from PIL import Image
import cv2
import pygame
from gtts import gTTS
import io

# Load the ViT model
model_name = 'Rajaram1996/FacialEmoRecog'
model = ViTForImageClassification.from_pretrained(model_name)

# Get the number of classes from the model configuration
num_classes = model.config.num_labels

# Update the emotions list based on the number of classes
emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

# Define preprocess for ViT model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize Pygame mixer for audio playback
pygame.mixer.init()

while True:
    # Capture a frame from the camera
    _, frame = camera.read()

    # Display the captured frame
    cv2.imshow("Real-Time Emotion Classification", frame)

    # Preprocess the frame for ViT
    image_data = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(image_data)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform emotion inference
    with torch.no_grad():
        outputs = model(input_batch)

    # Get the predicted emotion
    probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    predicted_emotion = emotions[predicted_class]

    # Convert the predicted emotion to speech
    tts = gTTS(predicted_emotion)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)

    # Play the audio file using Pygame
    audio_file.seek(0)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
