import gymnasium as gym
import logging
import numpy as np
from typing import Dict, List, Optional, Type, Union

import ray
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_bootstrap_value
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    global_norm,
    sequence_mask,
)
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class VTraceLoss:
    def __init__(
        self,
        actions,
        actions_logp,
        actions_entropy,
        dones,
        behaviour_action_logp,
        behaviour_logits,
        target_logits,
        discount,
        rewards,
        values,
        normalized_values,
        bootstrap_value,
        dist_class,
        model,
        valid_mask,
        config,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    ):
        """Policy gradient loss with vtrace importance weighting.

        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.

        Args:
            actions: An int|float32 tensor of shape [T, B, ACTION_SPACE].
            actions_logp: A float32 tensor of shape [T, B].
            actions_entropy: A float32 tensor of shape [T, B].
            dones: A bool tensor of shape [T, B].
            behaviour_action_logp: Tensor of shape [T, B].
            behaviour_logits: A list with length of ACTION_SPACE of float32
                tensors of shapes
                [T, B, ACTION_SPACE[0]],
                ...,
                [T, B, ACTION_SPACE[-1]]
            target_logits: A list with length of ACTION_SPACE of float32
                tensors of shapes
                [T, B, ACTION_SPACE[0]],
                ...,
                [T, B, ACTION_SPACE[-1]]
            discount: A float32 scalar.
            rewards: A float32 tensor of shape [T, B].
            values: A float32 tensor of shape [T, B].
            bootstrap_value: A float32 tensor of shape [B].
            dist_class: action distribution class for logits.
            valid_mask: A bool tensor of valid RNN input elements (#2992).
            config: Algorithm config dict.
        """


        import ray.rllib.algorithms.impala.vtrace_torch as vtrace

        if valid_mask is None:
            valid_mask = torch.ones_like(actions_logp)
        # Compute vtrace on the CPU for better perf
        # (devices handled inside `vtrace.multi_from_logits`).
        device = behaviour_action_logp[0].device
        self.vtrace_returns = vtrace.multi_from_logits(
            behaviour_action_log_probs=behaviour_action_logp,
            behaviour_policy_logits=behaviour_logits,
            target_policy_logits=target_logits,
            actions=torch.unbind(actions, dim=2),
            discounts=(1.0 - dones.float()) * discount,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            dist_class=dist_class,
            model=model,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold,
        )
        def valid_mean(value, valid_mask):
            return torch.sum(value * valid_mask) / torch.sum(valid_mask)

        with torch.no_grad():
            old_mu = model.mu
            old_sigma = model.sigma

            # update statistics

            G_value = self.vtrace_returns.vs.to(device)
            G_pg = self.vtrace_returns.pg_values.to(device)

            # new_max = torch.max(G_value)
            # new_min = torch.min(G_value)

            new_mu = valid_mean(G_value, valid_mask)
            new_nu = valid_mean(torch.pow(G_value, 2), valid_mask)
            new_xi = valid_mean(torch.pow(G_value, 3), valid_mask)
            new_omicron = valid_mean(torch.pow(G_value, 4), valid_mask)

            # model.new_g_max.copy_(new_max)
            # model.new_g_min.copy_(new_min)

            model.new_mu.copy_(new_mu)
            model.new_nu.copy_(new_nu)
            model.new_xi.copy_(new_xi)
            model.new_omicron.copy_(new_omicron)

            normalized_G_t_vtrace = torch.clip((G_value - old_mu) / old_sigma, -10.0, 10.0)
            normalized_G_t_pi = torch.clip((G_pg - old_mu) / old_sigma, -10.0, 10.0)

            pg_advantage = self.vtrace_returns.clipped_pg_rhos.to(device) * (normalized_G_t_pi - normalized_values)
        self.monitor_value = normalized_G_t_vtrace
        self.value_targets = self.vtrace_returns.vs.to(device)
        # The policy gradients loss.

        self.pi_loss = -torch.sum(
            actions_logp * pg_advantage.to(device) * valid_mask
        ) / torch.sum(valid_mask)

        # The baseline loss.
        delta = (normalized_G_t_vtrace - normalized_values)
        self.vf_loss = 0.5 * torch.sum(torch.pow(delta, 2.0) * valid_mask) / torch.sum(valid_mask)

        # The entropy loss.
        self.entropy = torch.sum(actions_entropy * valid_mask) / torch.sum(valid_mask)
        self.mean_entropy = self.entropy
        self.pg_loss = self.pi_loss + self.vf_loss * vf_loss_coeff
        convex_loss = self.pg_loss * (1+torch.pow(model.alpha*torch.exp(model.beta) - 1, 2)) \
                      + 0.1 * torch.pow(model.alpha, 2)
        # # The summed weighted loss.
        self.total_loss = (convex_loss - self.entropy * entropy_coeff)


def make_time_major(policy, seq_lens, tensor):
    """Swaps batch and trajectory axis.

    Args:
        policy: Policy reference
        seq_lens: Sequence lengths if recurrent or None
        tensor: A tensor or list of tensors to reshape.

    Returns:
        res: A tensor with swapped axes or a list of tensors with
        swapped axes.
    """
    if isinstance(tensor, (list, tuple)):
        return [make_time_major(policy, seq_lens, t) for t in tensor]

    if policy.is_recurrent():
        B = seq_lens.shape[0]
        T = tensor.shape[0] // B
    else:
        # Important: chop the tensor into batches at known episode cut
        # boundaries.
        # TODO: (sven) this is kind of a hack and won't work for
        #  batch_mode=complete_episodes.
        T = policy.config["rollout_fragment_length"]
        B = tensor.shape[0] // T

    rs = torch.reshape(tensor, [B, T] + list(tensor.shape[1:]))
    # Swap B and T axes.
    res = torch.transpose(rs, 1, 0)


    return res


class VTraceOptimizer:
    """Optimizer function for VTrace torch policies."""

    def __init__(self):
        pass

    @override(TorchPolicyV2)
    def optimizer(
        self,
    ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        if self.config["opt_type"] == "adam":
            return torch.optim.Adam(params=self.model.parameters(), lr=self.cur_lr)
        else:
            return torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.cur_lr,
                weight_decay=self.config["decay"],
                momentum=self.config["momentum"],
                eps=self.config["epsilon"],
            )


# VTrace mixins are placed in front of more general mixins to make sure
# their functions like optimizer() overrides all the other implementations
# (e.g., LearningRateSchedule.optimizer())
class ImpalaTorchPolicyCustom(
    VTraceOptimizer,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    ValueNetworkMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with Impala."""

    def __init__(self, observation_space, action_space, config):
        config = dict(
            ray.rllib.algorithms.impala.impala.ImpalaConfig().to_dict(), **config
        )

        # If Learner API is used, we don't need any loss-specific mixins.
        # However, we also would like to avoid creating special Policy-subclasses
        # for this as the entire Policy concept will soon not be used anymore with
        # the new Learner- and RLModule APIs.
        if not config.get("_enable_learner_api"):
            VTraceOptimizer.__init__(self)
            # Need to initialize learning rate variable before calling
            # TorchPolicyV2.__init__.
            LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
            EntropyCoeffSchedule.__init__(
                self, config["entropy_coeff"], config["entropy_coeff_schedule"]
            )

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)

        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        _ = model.update_popart()

        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)

        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        def _make_time_major(*args, **kw):
            return make_time_major(
                self, train_batch.get(SampleBatch.SEQ_LENS), *args, **kw
            )

        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_action_logp = train_batch[SampleBatch.ACTION_LOGP]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]
        if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
            unpacked_behaviour_logits = torch.split(
                behaviour_logits, list(output_hidden_shape), dim=1
            )
            unpacked_outputs = torch.split(model_out, list(output_hidden_shape), dim=1)
        else:
            unpacked_behaviour_logits = torch.chunk(
                behaviour_logits, output_hidden_shape, dim=1
            )
            unpacked_outputs = torch.chunk(model_out, output_hidden_shape, dim=1)

        values = model.value_function()
        values_time_major = _make_time_major(values)

        normalized_values = model.normalized_value_function()
        normalized_values_time_major = _make_time_major(normalized_values)

        bootstrap_values_time_major = _make_time_major(
            train_batch[SampleBatch.VALUES_BOOTSTRAPPED]
        )
        bootstrap_value = bootstrap_values_time_major[-1]

        if self.is_recurrent():
            max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = torch.reshape(mask_orig, [-1])
        else:
            mask = torch.ones_like(rewards)

        if mask.shape[0] < rewards.shape[0]:
            margin = rewards.shape[0] - mask.shape[0]
            pad = mask.new_zeros([margin]+list(mask.shape[1:]))
            mask = torch.cat([mask, pad],0)
        # Prepare actions for loss.
        loss_actions = actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)
        # Inputs are reshaped from [B * T] => [(T|T-1), B] for V-trace calc.
        loss = VTraceLoss(
            actions=_make_time_major(loss_actions),
            actions_logp=_make_time_major(action_dist.logp(actions)),
            actions_entropy=_make_time_major(action_dist.entropy()),
            dones=_make_time_major(dones),
            behaviour_action_logp=_make_time_major(behaviour_action_logp),
            behaviour_logits=_make_time_major(unpacked_behaviour_logits),
            target_logits=_make_time_major(unpacked_outputs),
            discount=self.config["gamma"],
            rewards=_make_time_major(rewards),
            values=values_time_major,
            normalized_values=normalized_values_time_major,
            bootstrap_value=bootstrap_value,
            dist_class=TorchCategorical if is_multidiscrete else dist_class,
            model=model,
            valid_mask=_make_time_major(mask),
            config=self.config,
            vf_loss_coeff=self.config["vf_loss_coeff"],
            entropy_coeff=self.entropy_coeff,
            clip_rho_threshold=self.config["vtrace_clip_rho_threshold"],
            clip_pg_rho_threshold=self.config["vtrace_clip_pg_rho_threshold"],
        )

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["pi_loss"] = loss.pi_loss
        model.tower_stats["vf_loss"] = loss.vf_loss
        model.tower_stats["entropy"] = loss.entropy
        model.tower_stats["mean_entropy"] = loss.mean_entropy
        model.tower_stats["total_loss"] = loss.total_loss
        model.tower_stats["monitor_value"] = loss.monitor_value

        values_batched = make_time_major(
            self,
            train_batch.get(SampleBatch.SEQ_LENS),
            values,
        )
        model.tower_stats["vf_explained_var"] = explained_variance(
            torch.reshape(loss.value_targets, [-1]), torch.reshape(values_batched, [-1])
        )

        return loss.total_loss

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        prev_a = train_batch[SampleBatch.ACTIONS]
        result_dict = {
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(torch.stack(self.get_tower_stats("pi_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "var_gnorm": global_norm(self.model.trainable_variables()),
                "vf_loss": torch.mean(torch.stack(self.get_tower_stats("vf_loss"))),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                )
            }

        result_dict["action/0"] = torch.mean((prev_a - 1) * (prev_a - 2.0) / 2)
        result_dict["action/1"] = -torch.mean(prev_a * (prev_a - 2.0))
        result_dict["action/2"] = torch.mean(prev_a * (prev_a - 1.0) / 2)

        result_dict["popart/mu"] = self.model.mu
        result_dict["popart/sigma"] = self.model.sigma
        result_dict["stat_3_skewness"] = self.model.skewness
        result_dict["stat_4_kurtosis"] = self.model.kurtosis
        # result_dict["popart/max"] = self.model.g_max
        # result_dict["popart/min"] = self.model.g_min

        result_dict['stat/alpha'] = self.model.alpha
        result_dict['stat/beta'] = self.model.beta

        # result_dict['norm/fc1'] = torch.norm(self.model.fc1.weight)
        # result_dict['norm/fc2'] = torch.norm(self.model.fc2.weight)

        result_dict['value/max'] = torch.max(torch.stack(self.get_tower_stats("monitor_value")))
        result_dict['value/min'] = torch.min(torch.stack(self.get_tower_stats("monitor_value")))

        return convert_to_numpy(result_dict
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[SampleBatch] = None,
        episode: Optional["Episode"] = None,
    ):
        # Call super's postprocess_trajectory first.
        # sample_batch = super().postprocess_trajectory(
        #    sample_batch, other_agent_batches, episode
        # )

        if self.config["vtrace"]:
            # Add the SampleBatch.VALUES_BOOTSTRAPPED column, which we'll need
            # inside the loss for vtrace calculations.
            sample_batch = compute_bootstrap_value(sample_batch, self)

        return sample_batch

    @override(TorchPolicyV2)
    def extra_grad_process(
        self, optimizer: "torch.optim.Optimizer", loss: TensorType
    ) -> Dict[str, TensorType]:
        return apply_grad_clipping(self, optimizer, loss)

    @override(TorchPolicyV2)
    def get_batch_divisibility_req(self) -> int:
        return self.config["rollout_fragment_length"]