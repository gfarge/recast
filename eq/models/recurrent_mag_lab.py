# TODO:
# - test magnitude sampling
# - Joint predictions with weighting: worse?
# - Large magnitude events
# - Magnitude prediction conditional on timing
# - Multicatalog training not working
# - test LSTM

# %%
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

import eq
import eq.distributions as dist


from pathlib import Path

project_root_path = Path(eq.__file__).parents[1]

print(project_root_path)

from eq.models.tpp_model import TPPModel


# %%
class RecurrentM2TPP(TPPModel):
    """Neural TPP model with an recurrent encoder.

    Args:
        input_magnitude: Should magnitude be used as model input?
        predict_magnitude: Should the model predict the magnitude?
        context_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
    """

    # TODO: add options for mag
    def __init__(
        self,
        predict_label: bool = True,
        input_label: bool = True,
        label_weight: float = 1.,
        predict_time: bool = True,
        input_time: bool = True,
        time_weight: float = 1.0,
        input_magnitude: bool = True,
        predict_magnitude: bool = True,
        magnitude_weight: float = 1.0,
        context_size: int = 32,
        num_components: int = 32,
        rnn_type: str = "GRU",
        dropout_proba: float = 0.5,
        tau_mean: float = 1.0,
        mag_mean: float = 0.0,
        richter_b: float = 1.0,
        mag_completeness: float = 2.0,
        learning_rate: float = 5e-2,
    ):
        super().__init__()
        self.input_label = input_label
        self.predict_label = predict_label
        self.input_time = input_time
        self.predict_time = predict_time
        self.input_magnitude = input_magnitude
        self.predict_magnitude = predict_magnitude
        self.context_size = context_size
        self.num_components = num_components
        self.register_buffer("tau_mean", torch.tensor(tau_mean, dtype=torch.float32))
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("mag_mean", torch.tensor(mag_mean, dtype=torch.float32))
        self.register_buffer("richter_b", torch.tensor(richter_b, dtype=torch.float32))
        self.register_buffer(
            "mag_completeness", torch.tensor(mag_completeness, dtype=torch.float32)
        )
        self.learning_rate = learning_rate

        self.logged_train_time_loss = []
        self.logged_train_mag_loss = []
        self.logged_train_lab_loss = []
        self.logged_val_time_loss = []
        self.logged_val_mag_loss = []
        self.logged_val_lab_loss = []

        loss_weights = []

        # Decoder for the time distribution
        if predict_time:
            self.num_time_params = 3 * self.num_components
            self.hypernet_time = nn.Linear(context_size, self.num_time_params)
            self.time_weight = time_weight
            loss_weights.append(time_weight)

        # Decoder for magnitude
        if predict_magnitude:
            self.num_mag_params = 3 * self.num_components
            self.hypernet_mag = nn.Linear(context_size, self.num_mag_params)
            self.magnitude_weight = magnitude_weight
            loss_weights.append(magnitude_weight)

        # Decoder for magnitude
        if predict_label:
            self.num_lab_params = 1
            self.hypernet_lab = nn.Linear(context_size, self.num_lab_params)
            self.label_weight = label_weight
            loss_weights.append(label_weight)


        self.loss_weights = torch.tensor(
            loss_weights, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        #self.time_magnitude_weight_ratio = time_weight / magnitude_weight

        # RNN input features
        if rnn_type not in ["RNN", "GRU"]:
            raise ValueError(
                f"rnn_type must be one of ['RNN', 'GRU'] " f"(got {rnn_type})"
            )
        self.num_rnn_inputs = int(input_time) + int(input_magnitude) + int(input_label)
        self.rnn = getattr(nn, rnn_type)(
            self.num_rnn_inputs, context_size, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_proba)

    def encode_time(self, inter_times):
        # inter_times has shape (...)
        # output has shape (..., 1)
        log_tau = torch.log(torch.clamp_min(inter_times, 1e-10)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def encode_magnitude(self, mag):
        # mag has shape (...)
        # output has shape (..., 1)
        return mag.unsqueeze(-1) - self.mag_mean

    def encode_label(self, labels):
        # labels has shape (...)
        # output has shape (..., 1)
        return labels.unsqueeze(-1)

    def get_context(self, batch):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        feat_list = []
        if self.input_label:
            feat_list.append(self.encode_label(batch.label))
        if self.input_time:
            feat_list.append(self.encode_time(batch.inter_times))
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        features = torch.cat(feat_list, dim=-1)

        rnn_output = self.rnn(features)[0][:, :-1, :]
        output = F.pad(rnn_output, (0, 0, 1, 0))  # (B, L, C)
        return self.dropout(output)  # (B, L, C)

    def get_label_dist(self, context):
        """Get the probability for the next label, given the context."""
        proba = self.hypernet_lab(context)
        print(proba)
        print(proba.shape)
        proba = F.softmax(proba.clamp_min(-5.), dim=0)
        print('after softmax')
        print(proba)
        print(proba.shape)

        return dist.Bernoulli(proba)

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        params = self.hypernet_time(context)
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
        scale, shape, weight_logits = torch.split(
            params,
            [self.num_components, self.num_components, self.num_components],
            dim=-1,
        )
        scale = F.softplus(scale.clamp_min(-5.0))
        shape = F.softplus(shape.clamp_min(-5.0))
        weight_logits = F.log_softmax(weight_logits, dim=-1)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)

        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def get_magnitude_dist(self, context):
        """Get the distribution over the magnitudes given the context."""

        params = self.hypernet_mag(context)

        scale, shape, weight_logits = torch.split(
            params,
            [self.num_components, self.num_components, self.num_components],
            dim=-1,
        )
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
        scale = F.softplus(scale.clamp_min(-5.0))
        shape = F.softplus(shape.clamp_min(-5.0))
        print('Weibull dist shape of shape parameter', shape.shape)

        weight_logits = F.log_softmax(weight_logits, dim=-1)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)

        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist, component_distribution=component_dist
        )

    def nll_loss(self, batch: eq.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        context = self.get_context(batch)  # (B, L, C)
        log_like_dict = {}


        # -----------------
        # Labels
        # -----------------
        if self.predict_label:
            label_dist = self.get_label_dist(context)
            log_pdf_label = label_dist.log_prob(batch.label)
            log_like_dict['nll_label'] = -(log_pdf_label * batch.mask).sum(-1) / (
                batch.t_end - batch.t_nll_start
            )

        # -----------------
        # Inter-event times
        # -----------------
        if self.predict_time is True:
            inter_time_dist = self.get_inter_time_dist(context)
            log_pdf_time = inter_time_dist.log_prob(
                batch.inter_times.clamp_min(1e-10)
            )  # (B, L)
            log_like_time = (log_pdf_time * batch.mask).sum(-1)

            # Survival time from last event until t_end
            arange = torch.arange(batch.batch_size)
            last_surv_context = context[arange, batch.end_idx, :]
            last_surv_dist = self.get_inter_time_dist(last_surv_context)
            last_log_surv = last_surv_dist.log_survival(
                batch.inter_times[arange, batch.end_idx]
            )
            log_like_time = log_like_time + last_log_surv.squeeze(-1)  # (B,)

            # Remove survival time from t_prev to t_nll_start
            if torch.any(batch.t_nll_start != batch.t_start):
                prev_surv_context = context[arange, batch.start_idx, :]
                prev_surv_dist = self.get_inter_time_dist(prev_surv_context)
                prev_surv_time = batch.inter_times[arange, batch.start_idx] - (
                    batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
                )
                prev_log_surv = prev_surv_dist.log_survival(prev_surv_time)
                log_like_time = log_like_time - prev_log_surv

            log_like_dict["nll_time"] = -log_like_time / (
                batch.t_end - batch.t_nll_start
            )

        # -----------------
        # Magnitudes
        # -----------------
        if self.predict_magnitude is True:
            magnitude_dist = self.get_magnitude_dist(context)
            log_pdf_mag = magnitude_dist.log_prob(
                batch.mag - self.mag_completeness
            )  # (B,L)
            log_like_dict["nll_mag"] = -(log_pdf_mag * batch.mask).sum(-1) / (
                batch.t_end - batch.t_nll_start
            )

        return log_like_dict

    def training_step(self, batch, batch_idx):
        nll_dict = self.nll_loss(batch)
        loss = (
               (int(self.input_time) * self.time_weight * nll_dict['nll_time']
               + int(self.input_magnitude) * self.magnitude_weight * nll_dict['nll_mag']
               + int(self.input_label) * self.magnitude_weight * nll_dict['nll_lab'])
               / (int(self.input_label)*self.lab_weight +
                   int(self.input_magnitude)*self.magnitude_weight +
                   int(self.input_time)*self.time_weight)  # w. average of NLLs
                ).mean()

        self.logged_train_time_loss.append(nll_dict['nll_time'].item())
        self.logged_train_mag_loss.append(nll_dict['nll_mag'].item())
        self.logged_train_lab_loss.append(nll_dict['nll_lab'].item())

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        self.log_dict(nll_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            nll_dict = self.nll_loss(batch)
            loss = (
                   (int(self.input_time) * self.time_weight * nll_dict['nll_time']
                   + int(self.input_magnitude) * self.magnitude_weight * nll_dict['nll_mag']
                   + int(self.input_label) * self.magnitude_weight * nll_dict['nll_lab'])
                   / (int(self.input_label)*self.lab_weight +
                       int(self.input_magnitude)*self.magnitude_weight +
                       int(self.input_time)*self.time_weight)  # w. average of NLLs
                    ).mean()

        self.logged_val_time_loss.append(nll_dict['nll_time'].item())
        self.logged_val_mag_loss.append(nll_dict['nll_mag'].item())
        self.logged_val_lab_loss.append(nll_dict['nll_lab'].item())

        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        self.log_dict(nll_dict)

    def test_step(self, batch, batch_idx, dataset_idx=None):
        with torch.no_grad():
            nll_dict = self.nll_loss(batch)
            loss = (
                   (int(self.input_time) * self.time_weight * nll_dict['nll_time']
                   + int(self.input_magnitude) * self.magnitude_weight * nll_dict['nll_mag']
                   + int(self.input_label) * self.magnitude_weight * nll_dict['nll_lab'])
                   / (int(self.input_label)*self.lab_weight +
                       int(self.input_magnitude)*self.magnitude_weight +
                       int(self.input_time)*self.time_weight)  # w. average of NLLs
                    ).mean()

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        self.log_dict(nll_dict)

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[eq.data.Sequence] = None,
        return_sequences: bool = False,
    ) -> Union[eq.data.Batch, List[eq.data.Sequence]]:
        """Simulate a batch of event sequences from the model.

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the interval on which to simulate the TPP.
            t_start: Start of the interval on which to simulate the TPP.
            past_seq: If provided, events are sampled conditioned on the past sequence.
            return_sequences: If True, returns samples as List[eq.data.Sequence].
                If False, returns samples as eq.data.Batch.

        Returns:
            batch: Sequences generated from the model.

        """
        if self.input_magnitude != self.predict_magnitude:
            raise ValueError(
                "Sampling is impossible if input_magnitude != predict_magnitude"
            )
        if past_seq is not None:
            t_start = past_seq.t_end
            past_batch = eq.data.Batch.from_list([past_seq])
            current_state = self.get_context(past_batch)[:, [-1], :]  # (1, 1, C)
            current_state = current_state.expand(batch_size, -1, -1)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
        else:
            current_state = torch.zeros(batch_size, 1, self.context_size)
            time_remaining = None
        t_end = t_start + duration

        inter_times = torch.empty(batch_size, 0, device=self.device)
        if self.predict_magnitude:
            magnitudes = torch.empty(batch_size, 0, device=self.device)
        else:
            magnitudes = None

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(current_state)
            if time_remaining is None:
                next_inter_times = inter_time_dist.sample()  # (B, 1)
            else:
                next_inter_times = inter_time_dist.sample_conditional(
                    lower_bound=time_remaining
                )  # (B, 1)
                next_inter_times -= time_remaining
            next_inter_times.clamp_max_(t_end - t_start)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (B, L)
            # Prepare RNN input
            rnn_input_list = [self.encode_time(next_inter_times)]

            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample() + self.mag_completeness  # (B, 1)
                magnitudes = torch.cat([magnitudes, next_mag], dim=1)  # (B, L)
                rnn_input_list.append(self.encode_magnitude(next_mag))

            with torch.no_grad():
                reached = inter_times.sum(-1).min()
                generated = reached >= t_end - t_start
                rnn_input = torch.cat(rnn_input_list, dim=-1)
            current_state = self.rnn(
                rnn_input, current_state.transpose(0, 1).contiguous()
            )[0]
            current_state = self.dropout(current_state)  # (B, 1, C)

        duration = t_end - t_start
        unclipped_arrival_times = inter_times.cumsum(-1)  # (B, L)
        padding_mask = unclipped_arrival_times > duration
        inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
        end_idx = (1 - padding_mask.long()).sum(-1)
        last_surv_time = duration - inter_times.sum(-1)
        inter_times[torch.arange(batch_size), end_idx] = last_surv_time
        batch = eq.data.Batch(
            inter_times=inter_times,
            arrival_times=inter_times.cumsum(-1),
            t_start=torch.full([batch_size], t_start, device=self.device).float(),
            t_end=torch.full([batch_size], t_end, device=self.device).float(),
            t_nll_start=torch.full([batch_size], t_start, device=self.device).float(),
            mask=padding_mask.float(),
            start_idx=torch.zeros(batch_size, device=self.device).long(),
            end_idx=end_idx,
            mag=magnitudes,
        )
        if return_sequences:
            return batch.to_list()
        else:
            return batch

    def evaluate_intensity(
        self,
        sequence: eq.data.Sequence,
        num_grid_points: int = 100,
        eps: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        inter_time_dist = self.get_inter_time_dist(context)

        # Evaluate each hazard function at times x = [eps, ..., tau_i]
        x = batch.inter_times * torch.linspace(eps, 1, num_grid_points)[:, None]
        intensity = inter_time_dist.log_hazard(x).T.reshape(-1).exp()

        # Shift the inter-event times x to get the global times
        offsets = torch.cat([torch.tensor([0.0]), sequence.arrival_times])
        grid = (x + offsets).T.reshape(-1)
        return grid, intensity

    def evaluate_compensator(
        self, sequence: eq.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = eq.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        inter_time_dist = self.get_inter_time_dist(context)

        # Evaluate each log survival function at times x = [eps, ..., tau_i]
        x = batch.inter_times * torch.linspace(1e-4, 1, num_grid_points)[:, None]
        log_surv = inter_time_dist.log_survival(x)
        # Compute the cumulative sum of log survival functions to get the compensator
        surv_offsets = torch.cat(
            [torch.tensor([0.0]), log_surv[-1].cumsum(dim=-1)[:-1]]
        )
        compensator = -(log_surv + surv_offsets).T.reshape(-1)

        # Shift the inter-event times x to get the global times
        offsets = torch.cat([torch.tensor([0.0]), sequence.arrival_times])
        grid = (x + offsets).T.reshape(-1)
        return grid, compensator
