
import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
import math

# paths
train_path = './dataset/features/training/'
test_path = './dataset/features/testing/'
demo_path = './dataset/features/testing/'
default_model_path = './demo_model/demo_model'
save_path = './demo_model/'
video_path = './dataset/videos/testing/positive/'

# batch_number (kept as default; code will auto-discover .npz files)
train_num = 126
test_num = 46

############## Train Parameters #################

# Parameters
learning_rate = 0.0001
n_epochs = 30
display_step = 10
batch_size = 10
# batch_size = 1

# Network Parameters
n_input = 4096
n_detection = 20
n_frames = 100
n_hidden = 512
# n_frames = 100
# n_hidden = 256
n_img_hidden = 256
n_att_hidden = 256
n_classes = 2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode', dest='mode', help='train, test or demo', default='demo')
    parser.add_argument('--model', dest='model', default=default_model_path)
    parser.add_argument('--gpu', dest='gpu', default='0')
    parser.add_argument('--video', dest='video', default=None, help='video id (filename without .mp4) to visualize in demo mode')
    parser.add_argument('--pad', dest='pad', default='none', choices=['none', 'pad'],
                        help="If 'pad', repeat last weight/bbox frame so visualization plays full video. Default 'none'.")
    args = parser.parse_args()

    return args


def build_model():
    # tf Graph input
    x = tf.placeholder("float", [None, n_frames, n_detection, n_input])
    y = tf.placeholder("float", [None, n_classes])
    keep = tf.placeholder("float", [None])

    # Define weights
    weights = {
        'em_obj': tf.Variable(tf.random_normal([n_input, n_att_hidden], mean=0.0, stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([n_input, n_img_hidden], mean=0.0, stddev=0.01)),
        'att_w': tf.Variable(tf.random_normal([n_att_hidden, 1], mean=0.0, stddev=0.01)),
        'att_wa': tf.Variable(tf.random_normal([n_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'att_ua': tf.Variable(tf.random_normal([n_att_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=0.0, stddev=0.01))
    }
    biases = {
        'em_obj': tf.Variable(tf.random_normal([n_att_hidden], mean=0.0, stddev=0.01)),
        'em_img': tf.Variable(tf.random_normal([n_img_hidden], mean=0.0, stddev=0.01)),
        'att_ba': tf.Variable(tf.zeros([n_att_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes], mean=0.0, stddev=0.01))
    }

    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden,
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                        use_peepholes=True,
                                        state_is_tuple=False)
    # using dropout in output of LSTM
    lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - keep[0])
    # init LSTM parameters (note: graph uses fixed batch_size variable)
    istate = tf.zeros([batch_size, lstm_cell.state_size])
    h_prev = tf.zeros([batch_size, n_hidden])
    # init loss
    loss = 0.0
    # Mask
    zeros_object = tf.cast(tf.not_equal(tf.reduce_sum(tf.transpose(x[:, :, 1:n_detection, :], [1, 2, 0, 3]), 3), 0),
                           dtype=tf.float32)  # frame x n x b
    # Start create graph
    for i in range(n_frames):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            # input features
            X = tf.transpose(x[:, i, :, :], [1, 0, 2])
            # frame embedded
            image = tf.matmul(X[0, :, :], weights['em_img']) + biases['em_img']
            # object embedded
            n_object = tf.reshape(X[1:n_detection, :, :], [-1, n_input])  # (n_steps*batch_size, n_input)
            n_object = tf.matmul(n_object, weights['em_obj']) + biases['em_obj']  # (n x b) x h
            n_object = tf.reshape(n_object, [n_detection - 1, batch_size, n_att_hidden])  # n-1 x b x h
            n_object = tf.multiply(n_object, tf.expand_dims(zeros_object[i], 2))

            # object attention
            brcst_w = tf.tile(tf.expand_dims(weights['att_w'], 0), [n_detection - 1, 1, 1])  # n x h x 1
            image_part = tf.matmul(n_object, tf.tile(tf.expand_dims(weights['att_ua'], 0), [n_detection - 1, 1, 1])) + biases[
                'att_ba']  # n x b x h
            e = tf.tanh(tf.matmul(h_prev, weights['att_wa']) + image_part)  # n x b x h
            # the probability of each object
            alphas = tf.multiply(tf.nn.softmax(tf.reduce_sum(tf.matmul(e, brcst_w), 2), 0), zeros_object[i])
            # weighting sum
            attention_list = tf.multiply(tf.expand_dims(alphas, 2), n_object)
            attention = tf.reduce_sum(attention_list, 0)  # b x h
            # concat frame & object
            fusion = tf.concat([image, attention], 1)

            with tf.variable_scope("LSTM") as vs:
                outputs, istate = lstm_cell_dropout(fusion, istate)
                lstm_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
            # save prev hidden state of LSTM
            h_prev = outputs
            # FC to output
            pred = tf.matmul(outputs, weights['out']) + biases['out']  # b x n_classes
            # save the predict of each time step
            if i == 0:
                soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred), (1, 0)), 1), (batch_size, 1))
                all_alphas = tf.expand_dims(alphas, 0)
            else:
                temp_soft_pred = tf.reshape(tf.gather(tf.transpose(tf.nn.softmax(pred), (1, 0)), 1), (batch_size, 1))
                soft_pred = tf.concat([soft_pred, temp_soft_pred], 1)
                temp_alphas = tf.expand_dims(alphas, 0)
                all_alphas = tf.concat([all_alphas, temp_alphas], 0)

            # positive example (exp_loss)
            pos_loss = -tf.multiply(tf.exp(-(n_frames - i - 1) / 20.0), -tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
            # negative example
            neg_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)  # Softmax loss

            temp_loss = tf.reduce_mean(tf.add(tf.multiply(pos_loss, y[:, 1]), tf.multiply(neg_loss, y[:, 0])))
            loss = tf.add(loss, temp_loss)

    # Define loss and optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss / n_frames)  # Adam Optimizer

    return x, keep, y, optimizer, loss, lstm_variables, soft_pred, all_alphas


def ensure_batch_size(data, labels, det, ID):
    """
    Ensure arrays have first dimension == batch_size.
    If fewer, repeat rows until we reach batch_size.
    If more, trim to batch_size.
    """
    n = data.shape[0]
    if n == batch_size:
        return data, labels, det, ID
    if n < batch_size:
        reps = math.ceil(batch_size / n)
        # tile then slice
        data_tiled = np.tile(data, (reps, 1, 1, 1))[:batch_size]
        labels_tiled = np.tile(labels, (reps, 1))[:batch_size]
        det_tiled = np.tile(det, (reps, 1, 1, 1))[:batch_size]
        ID_tiled = np.tile(ID, reps)[:batch_size]
        return data_tiled, labels_tiled, det_tiled, ID_tiled
    else:
        # n > batch_size -> trim (use first batch_size)
        data_trim = data[:batch_size]
        labels_trim = labels[:batch_size]
        det_trim = det[:batch_size]
        ID_trim = ID[:batch_size]
        return data_trim, labels_trim, det_trim, ID_trim


def train():
    # build model
    x, keep, y, optimizer, loss, lstm_variables, soft_pred, all_alphas = build_model()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    # mkdir folder for saving model
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=100)
    # Keep training until reach max iterations
    # start training
    for epoch in range(n_epochs):
        # random chose batch.npz
        epoch_loss = np.zeros((train_num, 1), dtype=float)
        n_batchs = np.arange(1, train_num + 1)
        np.random.shuffle(n_batchs)
        tStart_epoch = time.time()
        for batch in n_batchs:
            file_name = '%03d' % batch
            batch_fullpath = os.path.join(train_path, 'batch_' + file_name + '.npz')
            if not os.path.exists(batch_fullpath):
                print("Warning: training batch missing:", batch_fullpath)
                continue
            batch_data = np.load(batch_fullpath, allow_pickle=True)
            batch_xs = batch_data['data']
            batch_ys = batch_data['labels']
            # ensure first dimension == batch_size
            batch_xs, batch_ys, _, _ = ensure_batch_size(batch_xs, batch_ys, batch_data.get('det'), batch_data.get('ID'))
            _, batch_loss = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys, keep: [0.5]})
            epoch_loss[batch - 1] = batch_loss / batch_size

        print("Epoch:", epoch + 1, " done. Loss:", np.mean(epoch_loss))
        tStop_epoch = time.time()
        print("Epoch Time Cost:", round(tStop_epoch - tStart_epoch, 2), "s")
        sys.stdout.flush()
        if (epoch + 1) % 5 == 0:
            saver.save(sess, save_path + "model", global_step=epoch + 1)
            print("Saved checkpoint - Training")
            test_all(sess, train_num, train_path, x, keep, y, loss, lstm_variables, soft_pred)
            print("Testing")
            test_all(sess, test_num, test_path, x, keep, y, loss, lstm_variables, soft_pred)

    print("Optimization Finished!")
    saver.save(sess, save_path + "final_model")


def get_npz_files(path):
    """Return sorted list of batch_*.npz filenames (just the filename, not full path)."""
    if not os.path.isdir(path):
        return []
    files = [f for f in os.listdir(path) if f.endswith('.npz') and f.startswith('batch_')]
    files.sort()
    return files


def test_all(sess, num, path, x, keep, y, loss, lstm_variables, soft_pred):
    total_loss = 0.0
    # iterate using directory listing to avoid reliance on num
    npz_files = get_npz_files(path)
    if len(npz_files) == 0:
        print("No .npz files found in", path)
        return

    all_pred = None
    all_labels = None

    for npz_file in npz_files:
        npz_path = os.path.join(path, npz_file)
        try:
            test_all_data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print("Failed to load", npz_path, ":", e)
            continue
        test_data = test_all_data['data']
        test_labels = test_all_data['labels']
        # pad/trim to batch_size for graph compatibility
        test_data, test_labels, _, _ = ensure_batch_size(test_data, test_labels, test_all_data.get('det'), test_all_data.get('ID'))
        [temp_loss, pred] = sess.run([loss, soft_pred], feed_dict={x: test_data, y: test_labels, keep: [0.0]})
        total_loss += temp_loss / batch_size

        if all_pred is None:
            all_pred = pred[:, 0:90]
            all_labels = np.reshape(test_labels[:, 1], [batch_size, 1])
        else:
            all_pred = np.vstack((all_pred, pred[:, 0:90]))
            all_labels = np.vstack((all_labels, np.reshape(test_labels[:, 1], [batch_size, 1])))

    if all_pred is None:
        print("No predictions made (no valid batches).")
        return
    evaluation(all_pred, all_labels)


def evaluation(all_pred, all_labels, total_time=90, vis=False, length=None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident

    if length is not None:
        all_pred_tmp = np.zeros(all_pred.shape)
        for idx, vid in enumerate(length):
            all_pred_tmp[idx, total_time - vid:] = all_pred[idx, total_time - vid:]
        all_pred = np.array(all_pred_tmp)
        temp_shape = sum(length)
    else:
        length = [total_time] * all_pred.shape[0]
        temp_shape = all_pred.shape[0] * total_time
    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    for Th in sorted(all_pred.flatten()):
        if length is not None and Th == 0:
            continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(all_pred)):
            tp = np.where(all_pred[i] * all_labels[i] >= Th)
            Tp += float(len(tp[0]) > 0)
            if float(len(tp[0]) > 0) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter + 1
            Tp_Fp += float(len(np.where(all_pred[i] >= Th)[0]) > 0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp / Tp_Fp
        if np.sum(all_labels) == 0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp / np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (1 - time / counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _, rep_index = np.unique(Recall, return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    if len(new_Recall) == 0:
        print("No valid precision-recall points.")
        return

    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    print("Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " + "{:.4}".format(np.mean(new_Time) * 5))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)

    # handle case where recall array may not have 80% element
    if len(sort_recall) == 0:
        print("No recall values to compute Recall@80%.")
    else:
        idx80 = np.argmin(np.abs(sort_recall - 0.8))
        print("Recall@80%, Time to accident= " + "{:.4}".format(sort_time[idx80] * 5))


def _prepare_playback_and_pad_if_needed(new_weight, bboxes, cap, pad_mode):
    """
    Given arrays and an opened cv2.VideoCapture cap, optionally pad weight/bboxes
    if pad_mode == 'pad' and video has more frames than arrays.
    Returns updated new_weight, bboxes, and max_frames.
    """
    # shapes
    T_weight = new_weight.shape[0]
    T_bboxes = bboxes.shape[0]

    # try to get video frame count from capture
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frame_count <= 0:
        # fallback: unknown length -> use available weight length
        video_frame_count = max(T_weight, T_bboxes)

    max_frames = min(T_weight, T_bboxes)

    # if pad requested and video longer, pad arrays by repeating last frame
    if pad_mode == 'pad' and video_frame_count > max_frames:
        pad_count = video_frame_count - max_frames
        # pad new_weight: repeat last row
        last_w = new_weight[-1:, :].copy()
        pad_w = np.tile(last_w, (pad_count, 1))
        new_weight = np.vstack([new_weight, pad_w])
        # pad bboxes: repeat last bbox frame
        last_b = bboxes[-1:, :, :].copy()
        pad_b = np.tile(last_b, (pad_count, 1, 1))
        bboxes = np.vstack([bboxes, pad_b])
        max_frames = min(new_weight.shape[0], bboxes.shape[0])

    return new_weight, bboxes, max_frames


def vis(model_path, pad_mode='none'):
    # build model
    x, keep, y, optimizer, loss, lstm_variables, soft_pred, all_alphas = build_model()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    # restore model
    saver.restore(sess, model_path)
    # load data
    npz_files = get_npz_files(demo_path)
    if len(npz_files) == 0:
        print("No demo .npz files found in", demo_path)
        return

    for npz_file in npz_files:
        npz_path = os.path.join(demo_path, npz_file)
        all_data = np.load(npz_path, allow_pickle=True)
        data = all_data['data']
        labels = all_data['labels']
        det = all_data['det']
        ID = all_data['ID']

        # ensure shapes compatible with batch_size
        data, labels, det, ID = ensure_batch_size(data, labels, det, ID)

        # run result
        [all_loss, pred, weight] = sess.run([loss, soft_pred, all_alphas], feed_dict={x: data, y: labels, keep: [0.0]})

        for i in range(len(ID)):
            # We only visualize positive examples (accidents)
            if labels[i][1] == 1:
                print("\n--- Found a positive accident sample to visualize! ---")
                print("Step 1: Preparing to show the probability graph...")
                plt.figure(figsize=(14, 5))
                plt.plot(pred[i, 0:90], linewidth=3.0)
                plt.ylim(0, 1)
                plt.ylabel('Probability')
                plt.xlabel('Frame')
                plt.title('Probability curve')
                plt.show()
                print("Step 2: Graph window was closed. Proceeding to video...")

                file_name = ID[i].decode('utf-8') if isinstance(ID[i], bytes) else str(ID[i])
                full_video_path = os.path.join(video_path, file_name + '.mp4')
                print(f"Step 3: Attempting to open video file at: {full_video_path}")

                bboxes = det[i]
                new_weight = weight[:, :, i] * 255

                cap = cv2.VideoCapture(full_video_path)
                print(f"Step 4: Was the video file opened successfully? -> {cap.isOpened()}")
                if not cap.isOpened():
                    print("ERROR: OpenCV could not open the video file. This might be a video codec issue or wrong path.")
                    continue  # Skip to the next video

                ret, frame = cap.read()
                if frame is None:
                    print("ERROR: First frame is None. Skipping.")
                    cap.release()
                    continue

                # pad or compute safe max frames
                new_weight, bboxes, max_frames = _prepare_playback_and_pad_if_needed(new_weight, bboxes, cap, pad_mode)

                if max_frames == 0:
                    print("WARNING: no bbox/weight frames available for this sample. Skipping visualization.")
                    cap.release()
                    cv2.destroyAllWindows()
                    continue

                font = cv2.FONT_HERSHEY_SIMPLEX
                frame_counter = 0
                while ret and frame_counter < max_frames:
                    if frame_counter % 20 == 0:
                        print(f"  - Processing frame #{frame_counter}")

                    attention_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    now_weight = new_weight[frame_counter, :]
                    new_bboxes = bboxes[frame_counter, :, :]
                    index = np.argsort(now_weight)
                    for num_box in index:
                        x1 = int(max(0, min(frame.shape[1] - 1, new_bboxes[num_box, 0])))
                        y1 = int(max(0, min(frame.shape[0] - 1, new_bboxes[num_box, 1])))
                        x2 = int(max(0, min(frame.shape[1] - 1, new_bboxes[num_box, 2])))
                        y2 = int(max(0, min(frame.shape[0] - 1, new_bboxes[num_box, 3])))
                        if now_weight[num_box] / 255.0 > 0.4:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, str(round(now_weight[num_box] / 255.0 * 10000) / 10000),
                                    (x1, y1), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        attention_frame[y1:y2, x1:x2] = now_weight[num_box]

                    attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                    dst = cv2.addWeighted(frame, 0.6, attention_frame, 0.4, 0)
                    cv2.putText(dst, str(frame_counter + 1), (10, 30), font, 1, (255, 255, 255), 3)
                    cv2.imshow('result', dst)
                    c = cv2.waitKey(50) & 0xFF

                    ret, frame = cap.read()
                    frame_counter += 1

                    if (c == ord('q')) or (c == 27):
                        break

                if ret and frame_counter >= max_frames and pad_mode == 'none':
                    print(f"NOTE: stopped playback at frame {frame_counter} because only {max_frames} weight/bbox frames are available.")
                cap.release()
                cv2.destroyAllWindows()


def vis_specific_video(model_path, target_video_name, pad_mode='none'):
    """
    Find target_video_name in demo_path batch_*.npz files and visualize only that sample.
    target_video_name: filename without extension
    """
    x, keep, y, optimizer, loss, lstm_variables, soft_pred, all_alphas = build_model()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    print(f"Model restored from {model_path}. Looking for video '{target_video_name}' in {demo_path} .npz files...")

    found = False
    npz_files = get_npz_files(demo_path)
    for npz_file in npz_files:
        npz_path = os.path.join(demo_path, npz_file)
        try:
            all_data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print("Failed to load", npz_path, ":", e)
            continue

        IDs = all_data['ID']
        decoded_IDs = []
        for item in IDs:
            if isinstance(item, bytes):
                decoded_IDs.append(item.decode('utf-8'))
            else:
                decoded_IDs.append(str(item))

        # find indices with matching ID
        match_indices = [idx for idx, name in enumerate(decoded_IDs) if name == target_video_name]
        if len(match_indices) == 0:
            continue  # not in this batch file

        # We found the batch that contains the target video
        print(f"Found {target_video_name} in {npz_path} at indices {match_indices}")
        found = True

        data = all_data['data']
        labels = all_data['labels']
        det = all_data['det']
        ID = all_data['ID']

        # ensure shapes compatible with batch_size
        data, labels, det, ID = ensure_batch_size(data, labels, det, ID)

        # run model for the whole batch (model expects batch_size), get predictions and alphas
        [all_loss, pred, weight] = sess.run([loss, soft_pred, all_alphas], feed_dict={x: data, y: labels, keep: [0.0]})

        # for each matched index, visualize (usually just one)
        for i in match_indices:
            if labels[i][1] != 1:
                print(f"Warning: label for {decoded_IDs[i]} shows not-positive (labels[{i}] = {labels[i]}). Still visualizing.")

            print(f"\n--- Visualizing sample index {i} (ID={decoded_IDs[i]}) ---")
            plt.figure(figsize=(14, 5))
            plt.plot(pred[i, 0:90], linewidth=3.0)
            plt.ylim(0, 1)
            plt.ylabel('Probability')
            plt.xlabel('Frame')
            plt.title(f'Probability curve for {decoded_IDs[i]}')
            plt.show()
            plt.clf()

            file_name = decoded_IDs[i]
            full_video_path = os.path.join(video_path, file_name + '.mp4')
            print(f"Attempting to open {full_video_path}")
            bboxes = det[i]
            new_weight = weight[:, :, i] * 255

            cap = cv2.VideoCapture(full_video_path)
            if not cap.isOpened():
                print("ERROR: OpenCV could not open the video file. Check path / filename / codec.")
                continue

            ret, frame = cap.read()
            if frame is None:
                print("ERROR: First frame is None. Skipping.")
                cap.release()
                continue

            # pad or compute safe max frames
            new_weight, bboxes, max_frames = _prepare_playback_and_pad_if_needed(new_weight, bboxes, cap, pad_mode)

            if max_frames == 0:
                print("WARNING: no bbox/weight frames available for this sample. Skipping visualization.")
                cap.release()
                cv2.destroyAllWindows()
                continue

            frame_counter = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            while ret and frame_counter < max_frames:
                if frame_counter % 20 == 0:
                    print(f"  - Processing frame #{frame_counter}")
                attention_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                now_weight = new_weight[frame_counter, :]
                new_bboxes = bboxes[frame_counter, :, :]
                index = np.argsort(now_weight)
                for num_box in index:
                    x1 = int(max(0, min(frame.shape[1] - 1, new_bboxes[num_box, 0])))
                    y1 = int(max(0, min(frame.shape[0] - 1, new_bboxes[num_box, 1])))
                    x2 = int(max(0, min(frame.shape[1] - 1, new_bboxes[num_box, 2])))
                    y2 = int(max(0, min(frame.shape[0] - 1, new_bboxes[num_box, 3])))
                    if now_weight[num_box] / 255.0 > 0.4:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, str(round(now_weight[num_box] / 255.0 * 10000) / 10000),
                                (x1, y1), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    attention_frame[y1:y2, x1:x2] = now_weight[num_box]

                attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                dst = cv2.addWeighted(frame, 0.6, attention_frame, 0.4, 0)
                cv2.putText(dst, str(frame_counter + 1), (10, 30), font, 1, (255, 255, 255), 3)
                cv2.imshow('result', dst)
                c = cv2.waitKey(50) & 0xFF

                ret, frame = cap.read()
                frame_counter += 1

                if (c == ord('q')) or (c == 27):
                    break

            if ret and frame_counter >= max_frames and pad_mode == 'none':
                print(f"NOTE: stopped playback at frame {frame_counter} because only {max_frames} weight/bbox frames are available.")
            cap.release()
            cv2.destroyAllWindows()

        # done with found batch
        break

    if not found:
        print(f"Video {target_video_name} not found in any {demo_path} .npz file. Make sure demo_path's batches include this ID.")


def test(model_path):
    # load model
    x, keep, y, optimizer, loss, lstm_variables, soft_pred, all_alphas = build_model()
    # inistal Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    print("model restore!!!")
    print("Training")
    test_all(sess, train_num, train_path, x, keep, y, loss, lstm_variables, soft_pred)

    print("Testing")
    test_all(sess, test_num, test_path, x, keep, y, loss, lstm_variables, soft_pred)


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.mode == 'train':
        print("Starting training mode...")
        train()
    elif args.mode == 'test':
        print("Loading pre-trained model for testing...")
        test(args.model)
    elif args.mode == 'demo':
        print("Demo mode: visualizing predictions using pre-trained model...")
        if args.video:
            vis_specific_video(args.model, args.video, pad_mode=args.pad)
        else:
            vis(args.model, pad_mode=args.pad)
    else:
        print("Unknown mode. Use --mode train/test/demo")
