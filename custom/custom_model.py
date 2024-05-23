from typing import List

import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType

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
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space,
                         action_space,
                         num_outputs,
                         model_config,
                         name)
        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = model_config["custom_model_config"]["fc_size"]
        self.cell_size = model_config["custom_model_config"]["lstm_size"]
        self.hidden_size = model_config["custom_model_config"]["hidden_size"]

        self.action_space_struct = get_base_struct_from_space(self.action_space)
        self.action_dim = 0

        self.popart_beta = 1e-3

        lstm_input_size = self.obs_size

        # self.activation = nn.ReLU()
        self.activation = nn.SiLU()

        self.wrapped_fc1 = nn.Linear(self.obs_size, self.fc_size, bias=False)
        self.wrapped_ln1 = nn.LayerNorm(self.fc_size, eps=1e-8)
        self.wrapped_fc2 = nn.Linear(self.fc_size, self.obs_size, bias=False)
        self.wrapped_ln2 = nn.LayerNorm(self.obs_size, eps=1e-8)

        self.rnn = rnnlib.LRU(self.obs_size, self.cell_size, self.hidden_size, batch_first=True)
        # self.rnn1 = nn.GRU(lstm_input_size, self.cell_size, batch_first=True)
        # self.rnn2 = nn.GRU(self.cell_size, self.cell_size, batch_first=True)

        # self.rnn1 = rnnlib.LayerNormGRU(lstm_input_size, self.cell_size, batch_first=True)
        # self.rnn2 = rnnlib.LayerNormGRU(self.cell_size, self.cell_size, batch_first=True)

        self._logits_branch_sub = nn.Linear(self.cell_size, self.cell_size, bias=False)
        self._logits_branch_ln = nn.LayerNorm(self.cell_size, eps=1e-8)
        self._logits_branch = nn.Linear(self.cell_size, num_outputs)

        self._value_branch_sub = nn.Linear(self.cell_size, self.cell_size, bias=False)
        self._value_branch_ln = nn.LayerNorm(self.cell_size, eps=1e-8)
        self._value_branch = nn.Linear(self.cell_size, 1)

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
        with torch.no_grad():
            updated_mu = (1 - self.popart_beta) * self.mu + self.popart_beta * self.new_mu
            updated_nu = (1 - self.popart_beta) * self.nu + self.popart_beta * self.new_nu

            updated_sigma = torch.sqrt(updated_nu - torch.pow(updated_mu, 2))
            updated_sigma = torch.clamp(updated_sigma, 1e-6, 1e6)

            updated_xi = (1 - self.popart_beta) * self.xi + self.popart_beta * self.new_xi
            updated_omicron = (1 - self.popart_beta) * self.omicron + self.popart_beta * self.new_omicron

            moment3 = updated_xi - 3 * (updated_mu * updated_nu) + 2 * (torch.pow(updated_mu, 3))
            updated_skewness = moment3 / torch.pow(updated_sigma, 3)
            moment4 = updated_omicron - 4 * (updated_mu * updated_xi) + 6 * (
                        torch.pow(updated_mu, 2) * updated_nu) - 3 * (torch.pow(updated_mu, 4))
            updated_kurtosis = moment4 / torch.pow(updated_sigma, 4)

            self._value_branch.weight.copy_(self._value_branch.weight * self.sigma / updated_sigma)
            self._value_branch.bias.copy_((self._value_branch.bias * self.sigma + self.mu - updated_mu) / updated_sigma)

            self.mu.copy_(updated_mu)
            self.nu.copy_(updated_nu)
            self.xi.copy_(updated_xi)
            self.omicron.copy_(updated_omicron)

            self.sigma.copy_(updated_sigma)
            self.skewness.copy_(updated_skewness)
            self.kurtosis.copy_(updated_kurtosis)

        return self.mu, self.sigma

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens: TensorType, ):

        wrapped_out = input_dict["obs_flat"].float()
        skip_connection = wrapped_out

        wrapped_out = self.wrapped_fc1(wrapped_out)
        wrapped_out = self.wrapped_ln1(wrapped_out)
        wrapped_out = self.activation(wrapped_out)
        wrapped_out = self.wrapped_fc2(wrapped_out) + skip_connection
        wrapped_out = self.wrapped_ln2(wrapped_out)
        wrapped_out = self.activation(wrapped_out)

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

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

    # @override(TorchRNN)
    # def forward_rnn(
    #     self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    # ) -> (TensorType, List[TensorType]):
    #
    #     _h1 = torch.unsqueeze(state[0], 0)
    #     _h2 = torch.unsqueeze(state[1], 0)
    #
    #     net, h1_ = self.rnn1(inputs, _h1)
    #     self._features, h2_ = self.rnn2(net, _h2)
    #
    #     model_out = self._logits_branch_sub(self._features)
    #     model_out = self.activation(model_out)
    #     model_out = self._logits_branch(model_out)
    #
    #     return model_out, [torch.squeeze(h1_, 0), torch.squeeze(h2_, 0)]

    @override(TorchRNN)
    def forward_rnn(
            self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):

        _features, [h, c] = self.rnn(
            inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        self._features = _features + inputs
        model_out = self._logits_branch(self._features)
        return model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def get_initial_state(self) -> List[TensorType]:
        # Place hidden states on same device as model.
        h = [self._logits_branch.weight.new(
            1, self.hidden_size).zero_().squeeze(0),
             self._logits_branch.weight.new(
                 1, self.hidden_size).zero_().squeeze(0),
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
