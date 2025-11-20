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
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

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
VAL_SPLIT = 0.1
EVAL_EVERY = 5
GRAD_CLIP_NORM = 1.0
LR_FACTOR = 0.5
LR_PATIENCE = 2
SPLIT_SEED = 42
PSEUDO_TEST_SPLIT = 0.05

# Network Parameters
N_HIDDEN = 512
N_ATT_HIDDEN = 256
N_IMG_HIDDEN = 256
N_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True


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


def list_npz_files(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    return sorted([f for f in os.listdir(directory) if f.lower().endswith('.npz')])


def build_data_loaders(train_dir, val_split, split_seed, pseudo_test_split):
    files = list_npz_files(train_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz feature batches found in '{train_dir}'.")

    shuffled = files.copy()
    random.Random(split_seed).shuffle(shuffled)

    val_files = []
    if val_split > 0 and len(files) >= 2:
        val_count = max(1, int(len(shuffled) * val_split))
        val_files = sorted(shuffled[:val_count])
        remaining = shuffled[val_count:]
        print(f"Using {len(remaining)} batches for training and {len(val_files)} for validation (val_split={val_split:.2f}).")
    else:
        print("Validation split disabled or insufficient training files; training on all available batches.")
        remaining = shuffled

    pseudo_files = []
    if pseudo_test_split > 0 and len(remaining) > 1:
        pseudo_count = max(1, int(len(remaining) * pseudo_test_split))
        pseudo_count = min(pseudo_count, len(remaining) - 1)  # keep at least one batch for training
        pseudo_files = sorted(remaining[:pseudo_count])
        remaining = remaining[pseudo_count:]
        print(f"Reserving {len(pseudo_files)} batches from training for a temporary test set (pseudo_test_split={pseudo_test_split:.2f}).")

    train_files = sorted(remaining)
    if len(train_files) == 0:
        train_files = sorted(remaining + pseudo_files)
        pseudo_files = []
        print("Pseudo-test reservation consumed all training batches; reverting to train-only setup.")

    print(f"Training on {len(train_files)} batches after splits.")
    train_loader = create_dataloader(train_dir, shuffle=True, file_list=train_files)
    val_loader = create_dataloader(train_dir, shuffle=False, file_list=val_files) if val_files else None
    pseudo_loader = create_dataloader(train_dir, shuffle=False, file_list=pseudo_files) if pseudo_files else None
    return train_loader, val_loader, pseudo_loader


def evaluate_loss(model, data_loader):
    if data_loader is None:
        return None

    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for features, labels, _, _ in data_loader:
            b_features, b_labels, _, _ = ensure_batch_size(
                features.numpy(), labels.numpy(), np.zeros(1), np.zeros(1), BATCH_SIZE
            )
            b_features = torch.from_numpy(b_features).to(DEVICE)
            b_labels = torch.from_numpy(b_labels).to(DEVICE)

            preds, _ = model(b_features)
            loss = custom_loss_function(preds, b_labels)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def train(args):
    print("Starting training...")
    os.makedirs(SAVE_PATH, exist_ok=True)

    val_split = getattr(args, 'val_split', VAL_SPLIT)
    val_split = max(0.0, min(val_split, 0.5))
    eval_every = max(1, int(getattr(args, 'eval_every', EVAL_EVERY)))
    grad_clip = max(0.0, float(getattr(args, 'grad_clip', GRAD_CLIP_NORM)))
    split_seed = getattr(args, 'split_seed', SPLIT_SEED)
    lr_factor = getattr(args, 'lr_factor', LR_FACTOR)
    lr_patience = max(1, int(getattr(args, 'lr_patience', LR_PATIENCE)))
    pseudo_test_split = getattr(args, 'pseudo_test_split', PSEUDO_TEST_SPLIT)
    pseudo_test_split = max(0.0, min(pseudo_test_split, 0.5))
    use_amp = getattr(args, 'use_amp', True) and DEVICE.type == 'cuda'

    train_features_dir = os.path.join(DATASET_BASE, 'features/training')

    # Prepare test loader only if testing features exist (some users may only generate training .npz for now)
    test_loader = None
    test_features_dir = os.path.join(DATASET_BASE, 'features/testing')
    if os.path.isdir(test_features_dir):
        npz_files = [f for f in os.listdir(test_features_dir) if f.lower().endswith('.npz')]
        if len(npz_files) > 0:
            test_loader = create_dataloader(test_features_dir, shuffle=False)
        else:
            print(f"Warning: No .npz files found in testing features dir '{test_features_dir}'. Evaluation will be limited to training/validation.")
    else:
        print(f"Warning: Testing features directory '{test_features_dir}' does not exist. Evaluation will be limited to training/validation.")

    effective_pseudo_split = pseudo_test_split if test_loader is None else 0.0
    train_loader, val_loader, pseudo_test_loader = build_data_loaders(
        train_features_dir, val_split, split_seed, effective_pseudo_split
    )

    model = AccidentDetector(
        n_input=FEATURE_DIM, n_detection=N_DETECTION, n_frames=N_FRAMES,
        n_hidden=N_HIDDEN, n_att_hidden=N_ATT_HIDDEN, n_img_hidden=N_IMG_HIDDEN,
        n_classes=N_CLASSES, batch_size=BATCH_SIZE
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )
    scaler = GradScaler(enabled=use_amp)
    best_val_loss = float('inf')

    temp_test_loader = test_loader or pseudo_test_loader or None
    temp_test_name = "Testing Set" if test_loader is not None else None
    if temp_test_loader is None and val_loader is not None:
        temp_test_loader = val_loader
        temp_test_name = "Validation Set (temporary test)"
        print("Using validation batches as a temporary test set.")
    elif temp_test_loader is None:
        print("No data available for evaluation beyond training loss.")
    elif pseudo_test_loader is not None and temp_test_loader is pseudo_test_loader:
        temp_test_name = "Pseudo-Test Set"
        print("Using reserved training batches as a temporary test set.")

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

            with autocast(device_type=DEVICE.type, enabled=use_amp):
                preds, _ = model(b_features)
                loss = custom_loss_function(preds, b_labels)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        val_loss = None
        if val_loader is not None and ((epoch + 1) % eval_every == 0):
            val_loss = evaluate_loss(model, val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(SAVE_PATH, 'best_val_model.pth')
                torch.save(model.state_dict(), best_path)
                print(f"New best validation model saved to {best_path}")

        scheduler_metric = val_loss if val_loss is not None else avg_epoch_loss
        scheduler.step(scheduler_metric)

        if (epoch + 1) % eval_every == 0:
            checkpoint_path = os.path.join(SAVE_PATH, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

            if temp_test_loader is not None:
                print(f"--- Running Evaluation on {temp_test_name} ---")
                test_all(model, temp_test_loader, temp_test_name)
            else:
                print("No data available for evaluation beyond training loss.")

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

    if len(all_preds) == 0:
        print("No predictions collected during evaluation. Check that the dataloader is not empty or paths are correct.")
        return

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
    parser.add_argument('--val-split', type=float, default=VAL_SPLIT, help='Fraction of training .npz batches to reserve for validation (0 disables).')
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED, help='Random seed used for train/validation split.')
    parser.add_argument('--eval-every', type=int, default=EVAL_EVERY, help='How often (in epochs) to evaluate and checkpoint.')
    parser.add_argument('--grad-clip', type=float, default=GRAD_CLIP_NORM, help='Max gradient norm for clipping (<=0 disables).')
    parser.add_argument('--lr-factor', type=float, default=LR_FACTOR, help='Factor for ReduceLROnPlateau scheduler.')
    parser.add_argument('--lr-patience', type=int, default=LR_PATIENCE, help='Scheduler patience (eval steps) before reducing LR.')
    parser.add_argument('--pseudo-test-split', type=float, default=PSEUDO_TEST_SPLIT, help='Fraction of training batches reserved as temporary test data when no testing features exist.')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable mixed precision training (AMP).')
    parser.set_defaults(use_amp=True)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args.model)
    elif args.mode == 'demo':
        if not args.video:
            print("Please provide a video ID using --video <id> for demo mode.")
        else:
            vis(args.model, args.video)
    else:
        print("Unknown mode. Use 'train', 'test', or 'demo'.")