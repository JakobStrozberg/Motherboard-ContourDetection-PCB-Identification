import os
import glob
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from time import time
import supervision as sv

class ObjectDetection:

    def __init__(self, model_path):
        # Check if CUDA is available, otherwise use CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = YOLO(model_path)
        self.CLASS_NAMES_DICT = ['Button', 'Capacitor', 'Connector', 'Diode', 'Electrolytic Capacitor', 'IC', 'Inductor', 'Led', 'Pads', 'Pins', 'Resistor', 'Switch', 'Transistor']
        print(self.CLASS_NAMES_DICT)
    
        # Initialize the box annotator for visualizing object detections
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=1, text_thickness=1, text_scale=1)
    
    def plot_bboxes(self, results, frame): 
        # Extract detections from the YOLO model results
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        
        # Convert class IDs to integers
        class_id = class_id.astype(np.int32)
    
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=class_id,
                    )
    
        # Format custom labels for the object detections
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for xyxy, mask, confidence, class_id, track_id in detections]
        # Annotate and display the frame with the object detections
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    def process_and_save_images(self, images_paths, output_folder):
        # Loop through images
        for image_path in images_paths:
            # Read the image
            frame = cv2.imread(image_path)
      
            # Use the YOLO model to predict object detections in the frame
            results = self.model.predict(frame, conf=0.5)
            
            # Annotate the frame with the object detections
            frame = self.plot_bboxes(results, frame)
            
            # Save the annotated frame
            out_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(out_path, frame)

# Define paths
model_path = "/Users/jakobstrozberg/Documents/GitHub/AER850_Project_3/best150EPOCH.pt"
image_folder = "/Users/jakobstrozberg/Documents/GitHub/AER850_Project_3/image_folder"
output_folder = "/Users/jakobstrozberg/Documents/GitHub/AER850_Project_3/output_folder"

# Ensure the output folder is available
os.makedirs(output_folder, exist_ok=True)

# Get list of images
images_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# Initialize object detection
detector = ObjectDetection(model_path=model_path)

# Process and save images
detector.process_and_save_images(images_paths, output_folder)