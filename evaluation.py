# evaluation.py
import numpy as np

def evaluation(all_pred, all_labels, total_time=90):
    # This function is adapted from the original and works on numpy arrays.
    # all_pred: (N_videos, total_time) probabilities
    # all_labels: (N_videos, 1) ground truth (0 or 1)
    
    if all_pred.shape[0] != all_labels.shape[0]:
         raise ValueError("Shape mismatch between predictions and labels.")

    Precision = []
    Recall = []
    Time = []
    
    # Flatten all predictions and sort unique thresholds
    thresholds = sorted(list(np.unique(all_pred.flatten())))

    for Th in thresholds:
        if Th == 0: continue # Skip threshold 0

        # TP: video is positive and prediction > Th at some point
        # FP: video is negative and prediction > Th at some point
        # FN: video is positive and prediction <= Th everywhere
        
        Tp = 0
        Tp_Fp = 0 # Total predicted positives
        
        time_to_accident = 0
        
        for i in range(all_pred.shape[0]):
            # Positive prediction for this video if any frame prob > Th
            is_predicted_positive = np.any(all_pred[i, :] >= Th)
            
            if is_predicted_positive:
                Tp_Fp += 1
                if all_labels[i] == 1:
                    Tp += 1
                    # Time to accident calculation
                    first_detection_frame = np.where(all_pred[i, :] >= Th)[0][0]
                    # Original code seems to imply frames are from end of clip
                    # A 90-frame clip means accident is at frame 90
                    time_to_accident += (total_time - first_detection_frame)
        
        if Tp_Fp == 0:
            precision = 1.0 # Or np.nan, depending on desired behavior
        else:
            precision = Tp / Tp_Fp
        
        total_positives = np.sum(all_labels)
        if total_positives == 0:
            recall = np.nan
        else:
            recall = Tp / total_positives

        Precision.append(precision)
        Recall.append(recall)
        
        if Tp > 0:
            # Average time to accident for true positives at this threshold
            avg_tta = (time_to_accident / Tp) / 20.0 # Assuming 20 FPS, for seconds
            Time.append(avg_tta)
        else:
            Time.append(np.nan)

    # Calculate AP (Average Precision)
    Recall = np.array(Recall)
    Precision = np.array(Precision)
    
    # Sort by recall
    sorted_indices = np.argsort(Recall)
    Recall = Recall[sorted_indices]
    Precision = Precision[sorted_indices]
    
    # Use AUC for AP calculation
    AP = np.trapz(Precision, Recall)

    # Time to Accident (TTA)
    valid_times = [t for t in Time if not np.isnan(t)]
    mean_tta = np.mean(valid_times) if valid_times else 0.0

    print(f"Average Precision (AP): {AP:.4f}")
    print(f"Mean Time-to-Accident (TTA): {mean_tta:.4f}s")
    
    return AP, mean_tta