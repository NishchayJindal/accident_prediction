# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AccidentDetector(nn.Module):
    def __init__(self, n_input, n_detection, n_frames, n_hidden, n_att_hidden, n_img_hidden, n_classes, batch_size):
        super(AccidentDetector, self).__init__()
        self.n_input = n_input
        self.n_detection = n_detection
        self.n_frames = n_frames
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layers
        self.em_obj = nn.Linear(n_input, n_att_hidden)
        self.em_img = nn.Linear(n_input, n_img_hidden)

        # Attention mechanism weights
        self.att_w = nn.Linear(n_att_hidden, 1, bias=False)
        self.att_wa = nn.Linear(n_hidden, n_att_hidden, bias=False)
        self.att_ua = nn.Linear(n_att_hidden, n_att_hidden, bias=True)
        
        # LSTM
        self.lstm = nn.LSTMCell(n_img_hidden + n_att_hidden, n_hidden)
        self.dropout = nn.Dropout(p=0.5)

        # Output layer
        self.out = nn.Linear(n_hidden, n_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.lstm.weight_ih, mean=0.0, std=0.01)
        nn.init.normal_(self.lstm.weight_hh, mean=0.0, std=0.01)
        nn.init.constant_(self.lstm.bias_ih, 0)
        nn.init.constant_(self.lstm.bias_hh, 0)


    def forward(self, x):
        # x shape: (batch_size, n_frames, n_detection, n_input)
        
        # Initialize LSTM state
        h_prev = torch.zeros(self.batch_size, self.n_hidden, device=self.device)
        c_prev = torch.zeros(self.batch_size, self.n_hidden, device=self.device)
        
        all_preds = []
        all_alphas = []

        # Mask for zero-padded objects
        # Summing features across the last dimension to find non-zero objects
        object_mask = (x[:, :, 1:, :].sum(dim=-1) != 0).float() # (batch, frames, n_obj)

        for i in range(self.n_frames):
            # Input features for current frame
            frame_data = x[:, i, :, :]  # (batch, n_detection, n_input)
            
            # 1. Whole-frame feature
            img_feature = frame_data[:, 0, :]  # (batch, n_input)
            img_embedded = self.em_img(img_feature) # (batch, n_img_hidden)
            
            # 2. Object features
            obj_features = frame_data[:, 1:, :] # (batch, n_obj, n_input)
            obj_embedded = self.em_obj(obj_features) # (batch, n_obj, n_att_hidden)
            
            # Apply mask to zero out embeddings of padded objects
            obj_embedded = obj_embedded * object_mask[:, i, :].unsqueeze(-1)
            
            # 3. Attention Mechanism
            # e = tanh(h_prev * Wa + obj_emb * Ua + ba)
            e = torch.tanh(self.att_wa(h_prev).unsqueeze(1) + self.att_ua(obj_embedded)) # (batch, n_obj, n_att_hidden)
            
            # alphas = softmax(e * w)
            e = self.att_w(e).squeeze(-1) # (batch, n_obj)

            mask = (object_mask[:, i, :] == 0)
            if mask.any():
                min_val = torch.finfo(e.dtype).min
                e = e.masked_fill(mask, min_val)
            alphas = F.softmax(e, dim=1) # (batch, n_obj)
            all_alphas.append(alphas)

            # Weighted sum of object features
            attention = (alphas.unsqueeze(-1) * obj_embedded).sum(dim=1) # (batch, n_att_hidden)
            
            # 4. LSTM
            fusion = torch.cat([img_embedded, attention], dim=1) # (batch, n_img_hidden + n_att_hidden)
            h_prev, c_prev = self.lstm(fusion, (h_prev, c_prev))
            
            # Apply dropout
            outputs = self.dropout(h_prev)
            
            # 5. Output Prediction
            pred = self.out(outputs) # (batch, n_classes)
            all_preds.append(pred)

        # Stack results
        # final_preds shape: (batch_size, n_frames, n_classes)
        final_preds = torch.stack(all_preds, dim=1)
        # final_alphas shape: (batch_size, n_frames, n_obj)
        final_alphas = torch.stack(all_alphas, dim=1)

        return final_preds, final_alphas