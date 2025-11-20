# main.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import sys
import cv2
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import AccidentDetector
from dataset_loader import create_dataloader
from evaluation import evaluation

# --- Configuration ---
# Match these with feature_extractor.py
N_FRAMES = 100
N_OBJECTS = 19
FEATURE_DIM = 2048
N_DETECTION = N_OBJECTS + 1

# Paths
SAVE_PATH = './saved_models/'
DATASET_BASE = '/media/dev/Expansion/capstone/dataset/'

# Training Parameters
LEARNING_RATE = 0.0001
N_EPOCHS = 30
BATCH_SIZE = 10  # This is the batch size WITHIN one .npz file

# Network Parameters
N_HIDDEN = 512
N_ATT_HIDDEN = 256
N_IMG_HIDDEN = 256
N_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_loss_function(preds, labels):
    """
    Replicates the custom weighted loss from the original paper.
    preds: (batch, frames, classes)
    labels: (batch, classes) -> [1,0] for neg, [0,1] for pos
    """
    total_loss = 0.0
    for i in range(N_FRAMES):
        pred_at_t = preds[:, i, :]
        # Positive samples loss (accident)
        # Weight decreases as we get closer to the accident time
    # use math.exp to compute a float weight (torch.exp requires a tensor)
        pos_weight = math.exp(-(N_FRAMES - 1 - i) / 20.0)
        pos_loss = -pos_weight * F.log_softmax(pred_at_t, dim=1)[:, 1]

        # Negative samples loss (normal)
        neg_loss = -F.log_softmax(pred_at_t, dim=1)[:, 0]

        # Combine based on ground truth label
        loss_at_t = torch.mean(pos_loss * labels[:, 1] + neg_loss * labels[:, 0])
        total_loss += loss_at_t
    
    return total_loss / N_FRAMES

def ensure_batch_size(data, labels, det, ID, batch_size):
    n = data.shape[0]
    if n == batch_size:
        return data, labels, det, ID
    if n < batch_size:
        reps = math.ceil(batch_size / n)
        data = np.tile(data, (reps, 1, 1, 1))[:batch_size]
        labels = np.tile(labels, (reps, 1))[:batch_size]
        det = np.tile(det, (reps, 1, 1, 1))[:batch_size]
        ID = np.tile(ID, reps)[:batch_size]
    else:
        data = data[:batch_size]
        labels = labels[:batch_size]
        det = det[:batch_size]
        ID = ID[:batch_size]
    return data, labels, det, ID


def train():
    print("Starting training...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    model = AccidentDetector(
        n_input=FEATURE_DIM, n_detection=N_DETECTION, n_frames=N_FRAMES,
        n_hidden=N_HIDDEN, n_att_hidden=N_ATT_HIDDEN, n_img_hidden=N_IMG_HIDDEN,
        n_classes=N_CLASSES, batch_size=BATCH_SIZE
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = create_dataloader(os.path.join(DATASET_BASE, 'features/training'), shuffle=True)
    test_loader = create_dataloader(os.path.join(DATASET_BASE, 'features/testing'), shuffle=False)

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for batch_features, batch_labels, _, _ in progress_bar:
            
            b_features, b_labels, _, _ = ensure_batch_size(
                batch_features.numpy(), batch_labels.numpy(), np.zeros(1), np.zeros(1), BATCH_SIZE
            )
            b_features = torch.from_numpy(b_features).to(DEVICE)
            b_labels = torch.from_numpy(b_labels).to(DEVICE)

            optimizer.zero_grad()
            
            preds, _ = model(b_features)
            loss = custom_loss_function(preds, b_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        
        if (epoch + 1) % 5 == 0:
            print("--- Running Evaluation ---")
            test_all(model, test_loader, "Testing Set")
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"model_epoch_{epoch+1}.pth"))
            print(f"Saved model checkpoint to {SAVE_PATH}")
    
    print("Training finished.")
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "final_model.pth"))

def test_all(model, data_loader, set_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, _, _ in data_loader:
            b_features, b_labels, _, _ = ensure_batch_size(
                features.numpy(), labels.numpy(), np.zeros(1), np.zeros(1), BATCH_SIZE
            )
            b_features = torch.from_numpy(b_features).to(DEVICE)

            preds, _ = model(b_features)
            
            # Get probability of positive class (accident)
            probs = torch.softmax(preds, dim=-1)[:, :, 1].cpu().numpy()
            
            all_preds.append(probs)
            all_labels.append(b_labels[:, 1]) # Append 1 for positive, 0 for negative

    all_preds = np.vstack(all_preds)[:, :90] # Evaluate on first 90 frames as in original
    all_labels = np.hstack(all_labels).reshape(-1, 1)

    print(f"\n--- Evaluation on {set_name} ---")
    evaluation(all_preds, all_labels)

def test(model_path):
    print(f"Loading model from {model_path} for testing.")
    model = AccidentDetector(
        n_input=FEATURE_DIM, n_detection=N_DETECTION, n_frames=N_FRAMES,
        n_hidden=N_HIDDEN, n_att_hidden=N_ATT_HIDDEN, n_img_hidden=N_IMG_HIDDEN,
        n_classes=N_CLASSES, batch_size=BATCH_SIZE
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    test_loader = create_dataloader(os.path.join(DATASET_BASE, 'features/testing'), shuffle=False)
    test_all(model, test_loader, "Final Test Set")
    
def vis(model_path, video_id):
    print(f"Visualizing results for video: {video_id}")
    model = AccidentDetector(
        n_input=FEATURE_DIM, n_detection=N_DETECTION, n_frames=N_FRAMES,
        n_hidden=N_HIDDEN, n_att_hidden=N_ATT_HIDDEN, n_img_hidden=N_IMG_HIDDEN,
        n_classes=N_CLASSES, batch_size=BATCH_SIZE
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Find the video in the dataset files
    found = False
    demo_loader = create_dataloader(os.path.join(DATASET_BASE, 'features/testing'), shuffle=False)
    for features, labels, dets, ids in demo_loader:
        b_features, b_labels, b_dets, b_ids = ensure_batch_size(
            features.numpy(), labels.numpy(), dets, ids, BATCH_SIZE
        )
        
        decoded_ids = [s.decode('utf-8') for s in b_ids]
        if video_id not in decoded_ids:
            continue
        
        found = True
        sample_idx = decoded_ids.index(video_id)
        
        with torch.no_grad():
            b_features_tensor = torch.from_numpy(b_features).to(DEVICE)
            preds, alphas = model(b_features_tensor)
        
        # Extract results for the specific video
        probs = torch.softmax(preds, dim=-1)[sample_idx, :, 1].cpu().numpy()
        att_weights = alphas[sample_idx, :, :].cpu().numpy()
        bboxes = b_dets[sample_idx]
        
        # 1. Plot probability graph
        plt.figure(figsize=(14, 5))
        plt.plot(probs[:90], linewidth=3.0)
        plt.ylim(0, 1)
        plt.ylabel('Accident Probability')
        plt.xlabel('Frame')
        plt.title(f'Probability Curve for {video_id}')
        plt.show()

        # 2. Show video with attention overlay
        video_path = os.path.join(DATASET_BASE, 'videos/testing/positive', f'{video_id}.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(DATASET_BASE, 'videos/testing/negative', f'{video_id}.mp4')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video {video_path}")
            break

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_sample = np.linspace(0, frame_count - 1, N_FRAMES, dtype=int)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, frame_idx in enumerate(frames_to_sample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break

            attention_frame = np.zeros_like(frame, dtype=np.uint8)
            now_weights = att_weights[i, :]
            now_bboxes = bboxes[i, :, :]

            for box_idx in range(len(now_bboxes)):
                x1, y1, x2, y2 = map(int, now_bboxes[box_idx])
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0: continue # Skip padded boxes

                weight = now_weights[box_idx]
                color = (0, int(255 * (1-weight)), int(255 * weight)) # Green to Red
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{weight:.2f}", (x1, y1 - 5), font, 0.5, color, 1)

            # Display frame number and probability
            cv2.putText(frame, f"Frame: {i+1}", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Prob: {probs[i]:.3f}", (10, 70), font, 1, (0, 255, 0), 2)
            
            cv2.imshow('Accident Prediction Demo', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        break

    if not found:
        print(f"Video '{video_id}' not found in the testing feature set.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accident Prediction using PyTorch')
    parser.add_argument('--mode', dest='mode', help='train, test, or demo', default='demo')
    parser.add_argument('--model', dest='model', default=os.path.join(SAVE_PATH, 'final_model.pth'))
    parser.add_argument('--video', dest='video', default=None, help='Video ID (filename without .mp4) for demo mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test(args.model)
    elif args.mode == 'demo':
        if not args.video:
            print("Please provide a video ID using --video <id> for demo mode.")
        else:
            vis(args.model, args.video)
    else:
        print("Unknown mode. Use 'train', 'test', or 'demo'.")