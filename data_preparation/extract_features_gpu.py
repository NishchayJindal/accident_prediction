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
import gc

# --- Configuration ---
N_FRAMES = 100      # Number of frames to process per video clip
N_OBJECTS = 19      # Number of objects to consider
IMG_SIZE = 224      # Input size for the feature extractor (ResNet50)
FEATURE_DIM = 2048  # Output feature dimension from ResNet50

# --- Setup Models and Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load YOLOv8 for object detection
yolo_model = YOLO('yolov8l.pt').to(DEVICE)

# Load pre-trained ResNet50 for feature extraction
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
resnet_model.eval()

# Image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_in_batch(image_pil_list, model, batch_size=64):
    """
    OPTIMIZED: Extracts features from a list of PIL images in mini-batches.
    """
    if not image_pil_list:
        return np.array([])
        
    all_features = []
    with torch.inference_mode():
        for i in range(0, len(image_pil_list), batch_size):
            batch_pil = image_pil_list[i:i+batch_size]
            img_tensors = torch.stack([preprocess(img) for img in batch_pil]).to(DEVICE)
            features = model(img_tensors)
            all_features.append(features.squeeze(-1).squeeze(-1).cpu().numpy())
            
    return np.vstack(all_features)

def process_video(video_path, yolo_batch_size, yolo_img_size):
    """
    OPTIMIZED & MEMORY-RESILIENT: Processes a video by batching frames and objects,
    with auto-adjusting batch sizes to prevent OOM errors.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_sample_indices = np.linspace(0, frame_count - 1, N_FRAMES, dtype=int)
    
    frames_pil = []
    for frame_idx in frames_to_sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(frame_rgb))
    cap.release()
    
    sampled_frames_count = len(frames_pil)
    if sampled_frames_count == 0:
        print(f"Warning: No frames could be read from {video_path}")
        return None, None, None

    # --- Batch Object Detection with Auto-Adjusting Batch Size ---
    yolo_results = []
    current_yolo_batch_size = yolo_batch_size
    
    while current_yolo_batch_size > 0:
        try:
            # Process in mini-batches to fit into memory
            for i in tqdm(range(0, len(frames_pil), current_yolo_batch_size), desc="YOLO batch", leave=False):
                batch_frames = frames_pil[i:i+current_yolo_batch_size]
                results = yolo_model.predict(batch_frames, imgsz=yolo_img_size, verbose=False, device=DEVICE)
                yolo_results.extend(results)
            break # Success, exit the while loop
        except torch.cuda.OutOfMemoryError:
            print(f"\nWARNING: CUDA OOM with YOLO batch size {current_yolo_batch_size}. Halving and retrying.")
            current_yolo_batch_size //= 2
            yolo_results = [] # Reset results
            torch.cuda.empty_cache()
            gc.collect()
            if current_yolo_batch_size == 0:
                print("ERROR: YOLO batch size is zero. Cannot process video.")
                return None, None, None
    
    # --- Prepare "Mega-Batch" for Feature Extraction ---
    resnet_input_batch = []
    objects_per_frame_count = []
    all_bboxes_padded = []

    for i in range(sampled_frames_count):
        frame_pil = frames_pil[i]
        detections = yolo_results[i].boxes.data.cpu().numpy()

        if len(detections) > 0:
            detections = detections[detections[:, 4].argsort()[::-1]][:N_OBJECTS]
        
        num_found_objects = len(detections)
        objects_per_frame_count.append(num_found_objects)
        
        resnet_input_batch.append(frame_pil)
        
        frame_bboxes = []
        for det in detections:
            x1, y1, x2, y2, _, _ = det
            obj_img_pil = frame_pil.crop((x1, y1, x2, y2))
            # Ensure cropped image is not empty
            if obj_img_pil.width > 0 and obj_img_pil.height > 0:
                resnet_input_batch.append(obj_img_pil)
                frame_bboxes.append([x1, y1, x2, y2])
            else: # If crop is empty, decrement the found object count
                num_found_objects -= 1
        
        objects_per_frame_count[-1] = num_found_objects # Update with potentially corrected count
        
        pad_bboxes = np.zeros((N_OBJECTS - num_found_objects, 4))
        if num_found_objects > 0:
            all_bboxes_padded.append(np.vstack([np.array(frame_bboxes), pad_bboxes]))
        else:
            all_bboxes_padded.append(pad_bboxes)

    # --- Single Batch Feature Extraction ---
    all_extracted_features = extract_features_in_batch(resnet_input_batch, resnet_model)

    # --- Re-assemble Features ---
    all_features_structured = []
    feature_cursor = 0
    for i in range(sampled_frames_count):
        # The first feature is always the full frame
        frame_feature = all_extracted_features[feature_cursor]
        feature_cursor += 1
        
        num_objects = objects_per_frame_count[i]
        if num_objects > 0:
            object_features = all_extracted_features[feature_cursor : feature_cursor + num_objects]
            feature_cursor += num_objects
            pad_features = np.zeros((N_OBJECTS - num_objects, FEATURE_DIM))
            object_features = np.vstack([object_features, pad_features])
        else:
            object_features = np.zeros((N_OBJECTS, FEATURE_DIM))
            
        combined_features = np.vstack([frame_feature, object_features])
        all_features_structured.append(combined_features)
        
    # --- Pad Frames if video was shorter than N_FRAMES ---
    if sampled_frames_count < N_FRAMES:
        pad_count = N_FRAMES - sampled_frames_count
        last_feature = all_features_structured[-1]
        last_bbox = all_bboxes_padded[-1]
        all_features_structured.extend([last_feature] * pad_count)
        all_bboxes_padded.extend([last_bbox] * pad_count)

    return np.array(all_features_structured), np.array(all_bboxes_padded), os.path.splitext(os.path.basename(video_path))[0]


def main(args):
    print(f"Starting feature extraction for '{args.split}' split...")
    video_dir = os.path.join(args.base_path, 'videos', args.split)
    output_dir = os.path.join(args.base_path, 'features', args.split)
    os.makedirs(output_dir, exist_ok=True)
    
    all_video_paths, labels = [], []
    
    for label_type in ['positive', 'negative']:
        folder = os.path.join(video_dir, label_type)
        if not os.path.isdir(folder):
            print(f"Warning: Directory not found - {folder}")
            continue
        for video_file in os.listdir(folder):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                all_video_paths.append(os.path.join(folder, video_file))
                labels.append([1, 0] if label_type == 'negative' else [0, 1])

    batch_data, batch_labels, batch_det, batch_ID = [], [], [], []
    batch_counter = 1
    
    for i, video_path in enumerate(tqdm(all_video_paths, desc=f"Processing {args.split} videos")):
        features, bboxes, video_id = process_video(video_path, args.yolo_batch_size, args.yolo_img_size)
        
        if features is not None:
            batch_data.append(features)
            batch_labels.append(labels[i])
            batch_det.append(bboxes)
            batch_ID.append(video_id.encode('utf-8'))

            if len(batch_data) == args.batch_size or i == len(all_video_paths) - 1:
                if not batch_data: continue
                save_path = os.path.join(output_dir, f'batch_{batch_counter:03d}.npz')
                np.savez_compressed(save_path, data=np.array(batch_data), labels=np.array(batch_labels), det=np.array(batch_det), ID=np.array(batch_ID))
                print(f"\nSaved batch {batch_counter} to {save_path} with {len(batch_data)} videos")
                batch_data, batch_labels, batch_det, batch_ID = [], [], [], []
                batch_counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from videos.')
    parser.add_argument('--base_path', type=str, default='/media/dev/Expansion/capstone/dataset', help='Base path to dataset folder.')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'testing'], help='Dataset split to process.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of videos to group into one .npz file.')
    
    # --- NEW ARGUMENTS FOR MEMORY/SPEED CONTROL ---
    parser.add_argument('--yolo-batch-size', type=int, default=16, help='Batch size for YOLOv8 prediction to control VRAM usage.')
    parser.add_argument('--yolo-img-size', type=int, default=416, help='Input image size for YOLOv8 (e.g., 320, 416, 640). Smaller is faster and uses less VRAM.')

    args = parser.parse_args()
    main(args)