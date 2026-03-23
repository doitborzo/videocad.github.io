import torch
import torch.nn as nn
import transformers

from .trajectory_model import TrajectoryModel


class BaseTransformer(TrajectoryModel):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        enable_image_conditioning=True,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        n_layer=6,
        n_head=8,
        n_ctx=None,
        encoder="vit",
        use_pretrained_cad_model=False,
        **kwargs,
    ):
        super().__init__(state_dim, act_dim, encoder, use_pretrained_cad_model, max_length=max_length)

        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len

        if n_ctx is None:
            n_ctx = max_ep_len

        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_ctx=n_ctx,
            n_positions=n_ctx,
            **kwargs,
        )

        self.transformer = transformers.GPT2Model(config)
        self.enable_image_conditioning = enable_image_conditioning
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_embedding_model_size, hidden_size)
        self.embed_image = torch.nn.Linear(self.cad_embedding_model_size, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_action = nn.Sequential(*([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])))

    def create_simple_attention_mask(self, batch_size, seq_length, device):
        return torch.ones((batch_size, seq_length), dtype=torch.long, device=device)

    def create_attention_mask(self, batch_size, seq_length, device):
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
        image_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        stacked_attention_mask = torch.stack((attention_mask, attention_mask), dim=1).permute(0, 2, 1).reshape(
            batch_size, 2 * seq_length
        )
        if self.enable_image_conditioning:
            stacked_attention_mask = torch.cat((image_mask, stacked_attention_mask), dim=1)
        return stacked_attention_mask

    def get_transformer_outputs(self, states, actions, timesteps, cad_image, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            stacked_attention_mask = self.create_attention_mask(batch_size, seq_length, states.device)

        states_reshaped = states.view(-1, *states.shape[2:])
        state_embeddings = self.process_state(states_reshaped)
        state_embeddings = self.embed_state(state_embeddings)
        state_embeddings = state_embeddings.view(batch_size, seq_length, -1)

        image_embeddings = self.process_state(cad_image)
        image_embeddings = self.embed_state(image_embeddings)
        image_embeddings = image_embeddings.unsqueeze(1)

        actions_reshaped = actions.view(-1, self.act_dim)
        action_embeddings = self.embed_action(actions_reshaped)
        action_embeddings = action_embeddings.view(batch_size, seq_length, -1)

        state_embeddings = self.add_time_embeddings(state_embeddings, timesteps)
        action_embeddings = self.add_time_embeddings(action_embeddings, timesteps)

        stacked_inputs = self.stack_inputs(image_embeddings, state_embeddings, action_embeddings)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        return transformer_outputs

    def get_transformer_hidden_states(self, transformer_outputs, batch_size, seq_length):
        hidden_states = transformer_outputs["last_hidden_state"]
        if self.enable_image_conditioning:
            sequence_hidden = hidden_states[:, 1:]
        else:
            sequence_hidden = hidden_states
        sequence_hidden = sequence_hidden.reshape(batch_size, seq_length, 2, self.hidden_size)
        sequence_hidden = sequence_hidden.permute(0, 2, 1, 3)
        return sequence_hidden

    def add_time_embeddings(self, embeddings, timesteps):
        time_embeddings = self.embed_timestep(timesteps)
        return embeddings + time_embeddings

    def stack_inputs(self, cad_embeddings, state_embeddings, action_embeddings):
        batch_size = state_embeddings.shape[0]
        stacked = torch.stack((state_embeddings, action_embeddings), dim=1)
        stacked = stacked.permute(0, 2, 1, 3)
        stack = stacked.reshape(batch_size, -1, self.hidden_size)
        if self.enable_image_conditioning:
            return torch.cat((cad_embeddings, stack), dim=1)
        return stack

    def forward(self, states, actions, timesteps, cad_image, attention_mask=None):
        raise NotImplementedError("Subclasses should implement forward()")

    def get_action(self, states, actions, timesteps, **kwargs):
        _, action_preds = self.forward(states, actions, timesteps, **kwargs)
        return action_preds[:, -1]
