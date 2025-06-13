import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

from src.architecture.set_transformer import SetTransformer


class BrainDQN:
    def __init__(self, args, n_feat):
        self.args = args
        self.model = SetTransformer(
            dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=args.n_action
        )
        self.target_model = SetTransformer(
            dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=args.n_action
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.mode == "gpu":
            self.model.cuda()
            self.target_model.cuda()

    def train(self, x, y):
        self.model.train()
        set_input, _ = list(zip(*x))
        batched_set = torch.stack(set_input)
        y_pred = self.model(batched_set)
        y_tensor = torch.FloatTensor(np.array(y))
        if self.args.mode == "gpu":
            y_tensor = y_tensor.contiguous().cuda()
        loss = F.smooth_l1_loss(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, nn_input, target):
        with torch.no_grad():
            if target:
                self.target_model.eval()
                res = self.target_model(nn_input)
            else:
                self.model.eval()
                res = self.model(nn_input)
        return res.cpu().numpy()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder, filename):
        folder = os.path.normpath(folder)
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(folder):
        #     os.mkdir(folder)
        os.makedirs(folder, exist_ok=True)  # replace os.mkdir
        torch.save(self.model.state_dict(), filepath)
