import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.architecture.set_transformer import SetTransformer


class ActorCritic(nn.Module):
    def __init__(self, args, n_feat):
        super(ActorCritic, self).__init__()
        self.args = args
        self.action_layer = SetTransformer(
            dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=args.n_action
        )
        self.value_layer = SetTransformer(
            dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=1
        )

    def act(self, nn_input, available_tensor):
        if self.args.mode == "gpu":
            available_tensor = available_tensor.cuda()
        batched_nn_input = nn_input.unsqueeze(0)
        self.action_layer.eval()
        with torch.no_grad():
            out = self.action_layer(batched_nn_input)
            action_probs = out.squeeze(0)
            action_probs = action_probs + torch.abs(torch.min(action_probs))
            action_probs = action_probs - torch.max(action_probs * available_tensor)
            action_probs = self.masked_softmax(action_probs, available_tensor, dim=0)
            dist = Categorical(action_probs)
            action = dist.sample()
        return action, dist.log_prob(action), action_probs

    def evaluate(self, state_for_action, state_for_value, action, available_tensor):
        if self.args.mode == "gpu":
            available_tensor = available_tensor.cuda()
        action_probs = self.action_layer(state_for_action)
        action_probs = action_probs + torch.abs(
            torch.min(action_probs, 1, keepdim=True)[0]
        )
        action_probs = (
            action_probs
            - torch.max(action_probs * available_tensor, 1, keepdim=True)[0]
        )
        action_probs = self.masked_softmax(action_probs, available_tensor, dim=1)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state_for_value)
        return action_log_probs, torch.squeeze(state_value), dist_entropy

    @staticmethod
    def masked_softmax(vector, mask, dim=-1, temperature=1):
        mask_fill_value = -1e32
        memory_efficient = False
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                result = torch.nn.functional.softmax(
                    (vector / temperature) * mask, dim=dim
                )
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(
                    masked_vector / temperature, dim=dim
                )
        return result
