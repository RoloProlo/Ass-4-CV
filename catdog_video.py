import cv2
import torch
import numpy as np
from models import SmallObjectDetector
from evaluate import postprocess
from torchvision import transforms
from PIL import Image


IMG_SIZE = 112
S, C = 7, 2
OBJECTNESS_THRESHOLD = 0.5
CLASS_COLORS = {0: (0, 0, 255), 1: (255, 0, 0)}  # 0=cat (red), 1=dog (blue)
LABELS = {0: "Cat", 1: "Dog"}

# Load model
model = SmallObjectDetector()
model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device("cpu")))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Input and output video
input_path = "catdog.mp4"
output_path = "catdog_output.mp4"
cap = cv2.VideoCapture(input_path)

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened() and frame_count < 6000:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        detections = postprocess(prediction, threshold=OBJECTNESS_THRESHOLD)

    # Rescale boxes to original frame size
    scale_x = width / IMG_SIZE
    scale_y = height / IMG_SIZE

    for det in detections:
        label = det["class"]
        xmin, ymin, xmax, ymax = det["bbox"]
        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), CLASS_COLORS[label], 4)
        cv2.putText(frame, f"{LABELS[label]}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS[label], 2)

    # Frame counting
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(frame)
    frame_count += 1

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output saved to {output_path}")
