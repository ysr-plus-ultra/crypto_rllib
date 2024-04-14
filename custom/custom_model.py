import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import tree  # pip install dm_tree
from typing import Dict, List, Union
from ray.rllib.models.modelv2 import ModelV2

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
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

import rnnlib
torch, nn = try_import_torch()

class CustomRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=16,
        lstm_size=1024,
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

        self.popart_beta = 1e-3
        self.count = 1

        for space in tree.flatten(self.action_space_struct):
            if isinstance(space, Discrete):
                self.action_dim += space.n
            elif isinstance(space, MultiDiscrete):
                self.action_dim += np.sum(space.nvec)
            elif space.shape is not None:
                self.action_dim += int(np.product(space.shape))
            else:
                self.action_dim += int(len(space))



        if self.use_prev_action:
            self.fc_size += num_outputs
            self.obs_size += num_outputs
        if self.use_prev_reward:
            self.fc_size += 1
            self.obs_size += 1

        # self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        # self.fc2 = nn.Linear(self.fc_size, self.fc_size, bias=False)
        # self.ln2 = nn.LayerNorm(self.fc_size)
        #
        # self.activation = nn.ReLU6()
        # self.activation = nn.SiLU()

        self.rnn = rnnlib.LayerNormLSTM(self.obs_size, self.cell_size, batch_first=True)
        # self.rnn2 = rnnlib.LayerNormRNN(self.cell_size, self.cell_size, batch_first=True)
        # self.rnn1 = rnnlib.LayerNormLSTM(self.fc_size, self.cell_size, batch_first=True)
        # self.rnn2 = rnnlib.LayerNormLSTM(self.cell_size, self.fc_size, batch_first=True)
        # self.rnn = nn.LSTM(self.obs_size, self.cell_size, batch_first=True)

        self._logits_branch = nn.Linear(self.cell_size, num_outputs)
        self._value_branch = nn.Linear(self.cell_size, 1)

        # self.g_max = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # self.g_min = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.mu = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.nu = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.xi = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.omicron = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.skewness = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.kurtosis = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.new_g_max = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.new_g_min = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.new_mu = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.new_nu = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.new_xi = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.new_omicron = nn.Parameter(torch.tensor(1.0), requires_grad=False)

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


    def value_branch(self):
        assert self._features is not None, "must call forward() first"
        value_branch = self._value_branch(self._features)

        return value_branch

    @override(ModelV2)
    def value_function(self):
        normalized_output = self.value_branch()
        value_output = normalized_output * self.sigma + self.mu

        return torch.reshape(value_output, [-1])


    def normalized_value_function(self):
        normalized_output = self.value_branch()


        return torch.reshape(normalized_output, [-1])

    def update_popart(self):
        self.count += 1
        if self.count > 10000:
            adaptive_beta = self.popart_beta
        else:
            adaptive_beta = self.popart_beta / (1-(1-self.popart_beta)**self.count)

        with torch.no_grad():

            # updated_max = (1 - adaptive_beta) * self.g_max + adaptive_beta * self.new_g_max
            # updated_min = (1 - adaptive_beta) * self.g_min + adaptive_beta * self.new_g_min

            updated_mu = (1 - adaptive_beta) * self.mu + adaptive_beta * self.new_mu
            updated_nu = (1 - adaptive_beta) * self.nu + adaptive_beta * self.new_nu

            updated_sigma = torch.sqrt(updated_nu - torch.pow(updated_mu,2))
            updated_sigma = torch.clamp(updated_sigma, 1e-6, 1e6)

            # updated_mu = 0.5*(updated_max+updated_min)
            # updated_sigma = 0.5*(updated_max-updated_min)

            updated_xi = (1 - adaptive_beta) * self.xi + adaptive_beta * self.new_xi
            updated_omicron = (1 - adaptive_beta) * self.omicron + adaptive_beta * self.new_omicron

            moment3 = updated_xi - 3 * (updated_mu * updated_nu) + 2 * (torch.pow(updated_mu, 3))
            updated_skewness = moment3 / torch.pow(updated_sigma,3)
            moment4 = updated_omicron - 4 * (updated_mu * updated_xi) + 6 * (torch.pow(updated_mu, 2) * updated_nu) - 3 * (torch.pow(updated_mu, 4))
            updated_kurtosis = moment4 / torch.pow(updated_sigma,4)

            self._value_branch.weight *= self.sigma / updated_sigma
            self._value_branch.bias *= self.sigma / updated_sigma
            self._value_branch.bias += (self.mu - updated_mu) / updated_sigma

            self.mu.copy_(updated_mu)
            self.nu.copy_(updated_nu)
            self.xi.copy_(updated_xi)
            self.omicron.copy_(updated_omicron)

            self.sigma.copy_(updated_sigma)
            self.skewness.copy_(updated_skewness)
            self.kurtosis.copy_(updated_kurtosis)

            # self.g_max.copy_(updated_max)
            # self.g_min.copy_(updated_min)


        return self.mu, self.sigma

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens: TensorType,):
        float_input = input_dict["obs_flat"].float()
        wrapped_out = float_input

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
            wrapped_out = torch.cat([wrapped_out] + prev_a_r, dim=1)

        # layer
        # net = self.fc1(wrapped_out)
        # net = self.activation(net)
        # net = self.fc2(net)
        # wrapped_out = self.ln2(net)
        # wrapped_out = net + wrapped_out

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        self.time_major = self.model_config.get("_time_major", False)

        inputs = add_time_dimension(
            wrapped_out,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])

        return output, new_state

    # @override(TorchRNN)
    # def forward_rnn(
    #     self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    # ) -> (TensorType, List[TensorType]):
    #
    #     _h1 = torch.unsqueeze(state[0], 0)
    #     _h2 = torch.unsqueeze(state[1], 0)
    #     _h3 = torch.unsqueeze(state[2], 0)
    #     _h4 = torch.unsqueeze(state[3], 0)
    #
    #     net, [h1_, h2_] = self.rnn1(inputs, [_h1, _h2])
    #     features, [h3_, h4_] = self.rnn2(net, [_h3, _h4])
    #
    #     self._features = inputs + features
    #
    #     model_out = self._logits_branch(self._features)
    #
    #     return model_out, [torch.squeeze(h1_, 0), torch.squeeze(h2_, 0), torch.squeeze(h3_, 0), torch.squeeze(h4_, 0)]

    @override(TorchRNN)
    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):

        self._features, [h, c] = self.rnn(
            inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )

        model_out = self._logits_branch(self._features)
        return model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def get_initial_state(self) -> List[TensorType]:
        # Place hidden states on same device as model.
        h = [self.fc1.weight.new(
                1, self.cell_size).zero_().squeeze(0),
             self.fc1.weight.new(
                 1, self.cell_size).zero_().squeeze(0),
             ]
        return h

    # @override(ModelV2)
    # def get_initial_state(self) -> List[TensorType]:
    #     # Place hidden states on same device as model.
    #     h = [self.fc1.weight.new(
    #             1, self.cell_size).zero_().squeeze(0),
    #          self.fc1.weight.new(
    #              1, self.cell_size).zero_().squeeze(0),
    #          self.fc1.weight.new(
    #              1, self.fc_size).zero_().squeeze(0),
    #          self.fc1.weight.new(
    #              1, self.fc_size).zero_().squeeze(0),
    #          ]
    #     return h