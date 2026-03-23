import torch
import torch.nn as nn

from .base_transformer import BaseTransformer


class AutoRegressiveTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=1000,
        action_tanh=True,
        enable_past_actions=False,
        enable_past_states=False,
        enable_timestep_embedding=False,
        num_classes=5,
        num_params=6,
        num_params_values=1000,
        num_decoder_layers=8,
        dim_feedforward=512,
        use_pretrained_cad_model=False,
        nhead=4,
        dropout=0.1,
        normalize=False,
        device=None,
        encoder="vit",
        num_views=0,
        window_size=1,
        **kwargs,
    ):
        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            max_ep_len=max_ep_len,
            action_tanh=action_tanh,
            encoder=encoder,
            use_pretrained_cad_model=use_pretrained_cad_model,
            **kwargs,
        )
        self.enable_past_actions = enable_past_actions
        self.enable_past_states = enable_past_states
        self.act_dim = act_dim
        assert window_size > 0, "Window size must be greater than 0"
        self.window_size = window_size
        self.transformer_decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_decoder_layers,
        )
        self.normalize = normalize
        self.predict_action_class_0_4 = torch.nn.Linear(hidden_size, num_classes)
        self.predict_action_class_0_999 = torch.nn.Linear(hidden_size, num_params * num_params_values)
        self.num_views = num_views
        self.use_pretrained_cad_model = use_pretrained_cad_model

        self.num_inputs = 1
        if self.enable_past_states:
            self.num_inputs += 1
        if num_views > 0:
            self.embed_multiview = torch.nn.Linear(self.state_embedding_model_size * num_views, hidden_size)
            self.num_inputs += 1
        self.image_projection = torch.nn.Linear(hidden_size * self.num_inputs, hidden_size)
        self.embed_action = torch.nn.Linear(act_dim, hidden_size)
        self.enable_timestep_embedding = enable_timestep_embedding
        if self.enable_timestep_embedding:
            self.timestep_embedding = torch.nn.Embedding(max_ep_len, hidden_size)

        self.action_mask = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        ).float().to(device)

    def apply_action_mask(self, cmd_pred, param_pred):
        mask = self.action_mask[cmd_pred]
        masked_params = param_pred.clone()
        masked_params[mask == 0] = -1
        masked_params[:, :, 3] = torch.where(
            (masked_params[:, :, 2] >= 200) & (masked_params[:, :, 2] < 250),
            masked_params[:, :, 3],
            -1,
        )
        return masked_params

    def process_actions(self, actions):
        return self.embed_action(actions.float())

    def normalize_actions(self, actions):
        actions[:, :, 0] = actions[:, :, 0] / 4.0
        actions[:, :, 1:] = actions[:, :, 1:] / 1000.0
        return actions

    def forward(self, inputs, attention_mask=None):
        ui_images = inputs["frames"]
        actions = inputs["actions"]
        cad_image = inputs["cad_image"]
        multiview_images = inputs.get("multiview_images", None)

        batch_size, seq_length = actions.shape[0], actions.shape[1]
        timesteps = torch.arange(seq_length, device=actions.device)
        if self.enable_timestep_embedding:
            timesteps_embeddings = self.timestep_embedding(timesteps)
        else:
            timesteps_embeddings = torch.zeros(seq_length, self.hidden_size, device=actions.device)

        images = []
        if self.enable_past_states:
            ui_images_reshaped = ui_images.reshape(-1, *ui_images.shape[2:])
            ui_image_embeddings = self.process_state(ui_images_reshaped)
            ui_image_embeddings = self.embed_state(ui_image_embeddings).reshape(batch_size, seq_length, -1)
            ui_image_embeddings = ui_image_embeddings + timesteps_embeddings
            ui_image_embeddings = nn.Tanh()(ui_image_embeddings)
            if self.enable_past_actions:
                images.append(ui_image_embeddings)

        cad_image_embeddings = self.process_image(cad_image)
        cad_image_embeddings = self.embed_image(cad_image_embeddings).unsqueeze(1).repeat(1, seq_length, 1)
        images.append(cad_image_embeddings)

        if multiview_images is not None and self.num_views > 0:
            multiview_embeddings = self.process_multiview_images(multiview_images, seq_length)
            multiview_embeddings = self.embed_multiview(multiview_embeddings)
            images.append(multiview_embeddings)

        combined_image_embeddings = torch.cat(images, dim=-1)
        if len(images) > 1:
            combined_image_embeddings = self.image_projection(combined_image_embeddings)
        combined_image_embeddings = nn.Tanh()(combined_image_embeddings)
        action_embeddings = self.process_actions(actions)
        action_embeddings = action_embeddings + timesteps_embeddings
        action_embeddings = nn.Tanh()(action_embeddings)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length).to(cad_image.device)
        time_mask = torch.ones(seq_length, seq_length) * float(-torch.inf)
        rows = torch.arange(seq_length)[:, None]
        cols = torch.arange(seq_length)
        mask = (cols > (rows - self.window_size)) & (cols <= rows)
        time_mask[mask] = 0

        if self.enable_past_actions:
            transformer_outputs = self.transformer_decoder(
                tgt=action_embeddings.permute(1, 0, 2),
                memory=combined_image_embeddings.permute(1, 0, 2),
                tgt_mask=causal_mask.to(device=cad_image.device),
                memory_mask=time_mask.to(device=cad_image.device),
            )
        elif self.enable_past_states:
            transformer_outputs = self.transformer_decoder(
                tgt=ui_image_embeddings.permute(1, 0, 2),
                memory=combined_image_embeddings.permute(1, 0, 2),
                tgt_mask=time_mask.to(device=cad_image.device),
                memory_mask=time_mask.to(device=cad_image.device),
            )
        else:
            transformer_outputs = self.transformer_decoder(
                tgt=combined_image_embeddings.permute(1, 0, 2),
                memory=combined_image_embeddings.permute(1, 0, 2),
                tgt_mask=time_mask.to(device=cad_image.device),
                memory_mask=time_mask.to(device=cad_image.device),
            )
        sequence_hidden = transformer_outputs.permute(1, 0, 2)
        cmds = self.predict_action_class_0_4(sequence_hidden)
        params = self.predict_action_class_0_999(sequence_hidden).reshape(batch_size, seq_length, 6, 1000)
        return cmds, params
