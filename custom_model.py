import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
import tree  # pip install dm_tree
from typing import Dict, List, Union

from ray.rllib.models.modelv2 import ModelV2

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space

from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot


torch, nn = try_import_torch()

class CustomRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=6,
        lstm_size=6,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space,
                         action_space,
                         num_outputs,
                         model_config,
                         name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.cell_size = lstm_size

        self.use_prev_action = model_config["lstm_use_prev_action"]
        self.use_prev_reward = model_config["lstm_use_prev_reward"]

        self.action_space_struct = get_base_struct_from_space(self.action_space)
        self.action_dim = 0

        for space in tree.flatten(self.action_space_struct):
            if isinstance(space, Discrete):
                self.action_dim += space.n
            elif isinstance(space, MultiDiscrete):
                self.action_dim += np.sum(space.nvec)
            elif space.shape is not None:
                self.action_dim += int(np.product(space.shape))
            else:
                self.action_dim += int(len(space))

        self.fc1 = nn.Linear(self.obs_size, self.fc_size)


        if self.use_prev_action:
            self.fc_size += num_outputs
        if self.use_prev_reward:
            self.fc_size += 1

        self.norm_method = nn.BatchNorm1d(self.fc_size, momentum=0.01)
        self.rnn = nn.LSTM(self.fc_size, self.cell_size, batch_first=True)

        self._logits_branch = nn.Linear(self.cell_size, num_outputs)
        self._value_branch = nn.Linear(self.cell_size, 1)

        self.mu = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.nu = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.new_mu = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.new_nu = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        if model_config["lstm_use_prev_action"]:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
                SampleBatch.ACTIONS, space=self.action_space, shift=-1
            )
        if model_config["lstm_use_prev_reward"]:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
                SampleBatch.REWARDS, shift=-1
            )
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [self.fc1.weight.new(
                1, self.cell_size).zero_().squeeze(0),
             self.fc1.weight.new(
                 1, self.cell_size).zero_().squeeze(0)
             ]
        return h

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        normalized_output = self._value_branch(self._features)
        value_output = normalized_output * self.sigma + self.mu
        return torch.reshape(value_output, [-1])

    def normalized_value_function(self):
        assert self._features is not None, "must call forward() first"
        normalized_output = self._value_branch(self._features)
        return torch.reshape(normalized_output, [-1])

    def update_popart(self, beta):
        with torch.no_grad():
            updated_mu = (1 - beta) * self.mu + beta * self.new_mu
            updated_nu = (1 - beta) * self.nu + beta * self.new_nu

            updated_sigma = torch.clamp(torch.sqrt(updated_nu - (updated_mu ** 2)), 1e-6, 1e6)

            old_weight = self._value_branch.weight.data.clone()
            old_bias = self._value_branch.bias.data.clone()

            new_weight = (self.sigma / updated_sigma) * old_weight
            new_bias = (self.sigma * old_bias + self.mu - updated_mu) / updated_sigma

            self.mu.copy_(updated_mu)
            self.nu.copy_(updated_nu)
            self.sigma.copy_(updated_sigma)
            self._value_branch.weight.copy_(new_weight)
            self._value_branch.bias.copy_(new_bias)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        net = self.fc1(input_dict["obs_flat"].float())
        wrapped_out = nn.functional.relu(net)
        # Concat. prev-action/reward if required.
        prev_a_r = []
        # Prev actions.
        if self.model_config["lstm_use_prev_action"]:
            prev_a = input_dict[SampleBatch.PREV_ACTIONS]
            # If actions are not processed yet (in their original form as
            # have been sent to environment):
            # Flatten/one-hot into 1D array.
            if self.model_config["_disable_action_flattening"]:
                prev_a_r.append(
                    flatten_inputs_to_1d_tensor(
                        prev_a, spaces_struct=self.action_space_struct, time_axis=False
                    )
                )
            # If actions are already flattened (but not one-hot'd yet!),
            # one-hot discrete/multi-discrete actions here.
            else:
                if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                    prev_a = one_hot(prev_a.float(), self.action_space)
                else:
                    prev_a = prev_a.float()
                prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))
        # Prev rewards.
        if self.model_config["lstm_use_prev_reward"]:
            prev_a_r.append(
                torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(), [-1, 1])
            )
        # Concat prev. actions + rewards to the "main" input.
        if prev_a_r:
            wrapped_out = torch.cat([net] + prev_a_r, dim=1)

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = wrapped_out.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)

        wrapped_out = self.norm_method(wrapped_out)
        inputs = add_time_dimension(
            wrapped_out,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )


        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(TorchRNN)
    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):

        self._features, [h, c] = self.rnn(
            inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )

        model_out = self._logits_branch(self._features)
        return model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]