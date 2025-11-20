# data_preparation/extract_features.py
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import argparse

# --- Configuration ---
# You can adjust these parameters
N_FRAMES = 100      # Number of frames to process per video clip
N_OBJECTS = 19      # Number of objects to consider (k-1 in original code)
IMG_SIZE = 224      # Input size for the feature extractor (ResNet50)
FEATURE_DIM = 2048  # Output feature dimension from ResNet50's avgpool layer

# --- Setup Models and Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load YOLOv8 for object detection
yolo_model = YOLO('yolov8l.pt').to(DEVICE)

# Load pre-trained ResNet50 for feature extraction
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
# Remove the final classification layer to get feature embeddings
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
resnet_model.eval()

# Image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_from_image(image_pil, model):
    """Extracts features from a single PIL image."""
    with torch.no_grad():
        img_t = preprocess(image_pil).unsqueeze(0).to(DEVICE)
        features = model(img_t)
        return features.squeeze().cpu().numpy()

def process_video(video_path):
    """Processes a single video file to extract features and bounding boxes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_sample = np.linspace(0, frame_count - 1, N_FRAMES, dtype=int)

    all_features = []
    all_bboxes = []
    
    sampled_frames_count = 0
    for frame_idx in frames_to_sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        sampled_frames_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # 1. Extract whole-frame features
        frame_features = extract_features_from_image(frame_pil, resnet_model)

        # 2. Detect objects with YOLOv8
        results = yolo_model.predict(frame_pil, verbose=False)
        detections = results[0].boxes.data.cpu().numpy() # x1, y1, x2, y2, conf, cls

        # Sort by confidence and take top N_OBJECTS
        if len(detections) > 0:
            detections = detections[detections[:, 4].argsort()[::-1]][:N_OBJECTS]
        
        object_features = []
        object_bboxes = []

        for det in detections:
            x1, y1, x2, y2, _, _ = det
            # Crop object from the frame
            obj_img_pil = frame_pil.crop((x1, y1, x2, y2))
            
            # Extract features for the object
            obj_feat = extract_features_from_image(obj_img_pil, resnet_model)
            object_features.append(obj_feat)
            object_bboxes.append([x1, y1, x2, y2])
        
        # --- Padding ---
        # Pad object features and bboxes if fewer than N_OBJECTS were detected
        num_found_objects = len(object_features)
        
        # Pad features
        if num_found_objects < N_OBJECTS:
            pad_features = np.zeros((N_OBJECTS - num_found_objects, FEATURE_DIM))
            if num_found_objects > 0:
                object_features = np.vstack([np.array(object_features), pad_features])
            else:
                object_features = pad_features
        
        # Pad bboxes
        if num_found_objects < N_OBJECTS:
            pad_bboxes = np.zeros((N_OBJECTS - num_found_objects, 4))
            if num_found_objects > 0:
                object_bboxes = np.vstack([np.array(object_bboxes), pad_bboxes])
            else:
                object_bboxes = pad_bboxes
        
        # Combine frame feature with object features
        # Shape: (1 + N_OBJECTS, FEATURE_DIM) -> (n_detection, n_input)
        combined_features = np.vstack([frame_features, object_features])
        
        all_features.append(combined_features)
        all_bboxes.append(np.array(object_bboxes))
        
    cap.release()
    
    # Pad frames if video was shorter than N_FRAMES
    if sampled_frames_count < N_FRAMES:
        pad_count = N_FRAMES - sampled_frames_count
        if sampled_frames_count > 0:
            last_feature = all_features[-1]
            last_bbox = all_bboxes[-1]
            all_features.extend([last_feature] * pad_count)
            all_bboxes.extend([last_bbox] * pad_count)
        else: # Video was empty or unreadable
            return None, None, None

    return np.array(all_features), np.array(all_bboxes), os.path.splitext(os.path.basename(video_path))[0]


def main(args):
    print("Starting feature extraction...")
    video_dir = os.path.join(args.base_path, 'videos', args.split)
    output_dir = os.path.join(args.base_path, 'features', args.split)
    os.makedirs(output_dir, exist_ok=True)
    
    all_video_paths = []
    labels = []
    
    # Collect all video paths and assign labels
    for label_type in ['positive', 'negative']:
        folder = os.path.join(video_dir, label_type)
        if not os.path.isdir(folder):
            continue
        for video_file in os.listdir(folder):
            if video_file.endswith('.mp4'):
                all_video_paths.append(os.path.join(folder, video_file))
                # Label: [1, 0] for negative, [0, 1] for positive
                labels.append([1, 0] if label_type == 'negative' else [0, 1])

    batch_data, batch_labels, batch_det, batch_ID = [], [], [], []
    batch_counter = 1
    
    for i, video_path in enumerate(tqdm(all_video_paths, desc=f"Processing {args.split} videos")):
        features, bboxes, video_id = process_video(video_path)
        
        if features is not None:
            batch_data.append(features)
            batch_labels.append(labels[i])
            batch_det.append(bboxes)
            batch_ID.append(video_id.encode('utf-8')) # Match original format

            if len(batch_data) == args.batch_size or i == len(all_video_paths) - 1:
                save_path = os.path.join(output_dir, f'batch_{batch_counter:03d}.npz')
                np.savez_compressed(
                    save_path,
                    data=np.array(batch_data),
                    labels=np.array(batch_labels),
                    det=np.array(batch_det),
                    ID=np.array(batch_ID)
                )
                print(f"Saved batch {batch_counter} to {save_path}")
                batch_data, batch_labels, batch_det, batch_ID = [], [], [], []
                batch_counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from videos.')
    parser.add_argument('--base_path', type=str, default='/media/dev/Expansion/capstone/dataset', help='Base path to dataset folder.')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'testing'], help='Dataset split to process.')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of videos to group into one .npz file.')
    args = parser.parse_args()
    main(args)