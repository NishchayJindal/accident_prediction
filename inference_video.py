"""inference_video.py

Run inference on a single MP4, produce:
 - a probability-vs-frame plot (saved as PNG)
 - a labeled output video with bounding boxes colored by attention and frames annotated with probability

This script re-uses the project's `AccidentDetector` model and the feature extraction helper
`process_video` from `data_preparation/extract_features_gpu.py`.

Usage examples:
python inference_video.py --model ./saved_models/final_model.pth --video /path/to/video.mp4 --out_dir ./outputs

Notes:
- Extracting features requires YOLOv8 and a ResNet model; importing the extractor will initialize those models and may take time/VRAM.
- The script saves outputs to --out_dir (created if missing).
"""

import os
import argparse
import math
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from model import AccidentDetector
from data_preparation.extract_features_gpu import process_video, N_FRAMES, N_OBJECTS, FEATURE_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(model_path, device=DEVICE):
    model = AccidentDetector(
        n_input=FEATURE_DIM, n_detection=N_OBJECTS + 1, n_frames=N_FRAMES,
        n_hidden=512, n_att_hidden=256, n_img_hidden=256,
        n_classes=2, batch_size=1
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def run_inference_on_video(model, video_path, out_dir, device=DEVICE, prob_threshold=0.5, yolo_batch_size=8, yolo_img_size=416):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"Extracting features from {video_path} (this will initialize YOLO/ResNet if not already)...")
    features, bboxes, vid_id = process_video(video_path, yolo_batch_size, yolo_img_size)
    if features is None:
        raise RuntimeError("Feature extraction failed for the provided video.")

    # features: (N_FRAMES, N_DETECTION, FEATURE_DIM)
    # bboxes: (N_FRAMES, N_OBJECTS, 4)
    # Build batch dim
    batch_feat = np.expand_dims(features, axis=0).astype(np.float32)
    with torch.no_grad():
        feats_t = torch.from_numpy(batch_feat).to(device)
        preds, alphas = model(feats_t)
        probs = torch.softmax(preds, dim=-1).cpu().numpy()[0, :, 1]  # (N_FRAMES,)
        att_weights = alphas.cpu().numpy()[0]  # (N_FRAMES, N_DETECTION?)

    # Save probability plot
    plt.figure(figsize=(12, 4))
    plt.plot(probs[:90], linewidth=2)
    plt.ylim(0, 1)
    plt.xlabel('Frame (sampled)')
    plt.ylabel('Accident probability')
    plt.title(f'Accident probability curve - {os.path.basename(video_path)}')
    plot_path = os.path.join(out_dir, f"{vid_id}_prob_curve.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved probability curve to {plot_path}")

    # Prepare to write labeled video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path} for annotation")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(out_dir, f"{vid_id}_labeled.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frames_to_sample = np.linspace(0, frame_count - 1, N_FRAMES, dtype=int)
    frames_to_sample_set = set(frames_to_sample.tolist())

    font = cv2.FONT_HERSHEY_SIMPLEX
    sample_idx = 0
    frame_idx = 0

    print(f"Writing labeled video to {out_video_path} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frames_to_sample_set:
            i = sample_idx  # index into features/bboxes/probs
            # overlay bboxes and attention
            now_bboxes = bboxes[i]
            # att_weights could be sized (N_FRAMES, N_DETECTION) where N_DETECTION==N_OBJECTS+1
            now_weights = att_weights[i]

            # Draw boxes
            for box_idx in range(len(now_bboxes)):
                x1, y1, x2, y2 = now_bboxes[box_idx]
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                # Skip zero boxes
                if x1_i == 0 and y1_i == 0 and x2_i == 0 and y2_i == 0:
                    continue

                # Weight index: if att has same length as detection dim, use box_idx+1? The model likely uses first entry for full frame
                # We assume att_weights correspond to detections including the full-frame feature first; therefore object idx maps to box_idx+1
                att_idx = min(box_idx + 1, now_weights.shape[0]-1)
                weight = float(now_weights[att_idx]) if now_weights.size > att_idx else 0.0

                # color from green->red
                color = (0, int(255 * (1 - weight)), int(255 * weight))
                thickness = 2
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, thickness)
                cv2.putText(frame, f"{weight:.2f}", (x1_i, max(0, y1_i - 6)), font, 0.5, color, 1)

            # Add probability text
            prob = probs[i]
            label = f"Prob: {prob:.3f}"
            cv2.putText(frame, label, (10, 30), font, 1.0, (255, 255, 255), 2)

            # If above threshold, add red banner
            if prob >= prob_threshold:
                cv2.putText(frame, "PREDICTED ACCIDENT", (10, 70), font, 1.0, (0, 0, 255), 3)

            sample_idx += 1

        # write frame (whether sampled or not)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Labeled video saved to {out_video_path} (took {time.time()-t0:.1f}s)")
    return plot_path, out_video_path


def main():
    parser = argparse.ArgumentParser(description='Run inference on a standalone mp4 and output plot + labeled video')
    parser.add_argument('--model',default="./saved_models/best_val_model.pth", required=False, help='Path to saved model .pth')
    parser.add_argument('--video',default=r"C:\Users\Dell\Desktop\cap\accident-prediction\video_path\WhatsApp Video 2025-11-19 at 22.15.43_5ab47800.mp4", required=False, help='Path to mp4 file to run inference on')
    parser.add_argument('--out_dir', default='./outputs', help='Output directory for plot and labeled video')
    parser.add_argument('--device', default=None, help='torch device to use, e.g. cuda or cpu (overrides auto)')
    parser.add_argument('--prob-threshold', type=float, default=0.5, help='Threshold to mark frames as predicted-accident')
    parser.add_argument('--yolo-batch-size', type=int, default=8, help='Batch size used internally by the extractor')
    parser.add_argument('--yolo-img-size', type=int, default=416, help='YOLO input image size')

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = DEVICE

    model = load_model(args.model, device)
    plot_path, video_path = run_inference_on_video(model, args.video, args.out_dir, device=device, prob_threshold=args.prob_threshold, yolo_batch_size=args.yolo_batch_size, yolo_img_size=args.yolo_img_size)

    print('Done.')
    print(f'Plot: {plot_path}')
    print(f'Labeled video: {video_path}')


if __name__ == '__main__':
    main()
