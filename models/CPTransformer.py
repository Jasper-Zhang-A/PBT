import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PositionalEmbedding_BL
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        configs = configs.ec_config.get_configs()
        self.d_ff = configs.d_ff
        self.d_llm = configs.d_llm
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.use_agent = getattr(configs, 'use_agent', False)
        self.num_total_experts = (
            getattr(configs, 'cathode_experts', 0)
            + getattr(configs, 'temperature_experts', 0)
            + getattr(configs, 'format_experts', 0)
            + getattr(configs, 'anode_experts', 0)
        )
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(self.charge_discharge_length*3, self.d_model)
        self.intra_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.e_layers)])

        self.pe = PositionalEmbedding_BL(self.d_model)
        self.inter_TransformerEncoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ]
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.inter_flatten = nn.Flatten(start_dim=1)
        self.flat_dim = configs.d_model * self.early_cycle_threshold
        self.projection = nn.Linear(self.flat_dim, configs.output_num)
        if self.use_agent:
            self.agent_projector = nn.Sequential(
                nn.Linear(self.d_llm + 1, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.agent_fusion_mlp = nn.Sequential(
                nn.Linear(self.flat_dim + self.d_model, self.flat_dim),
                nn.GELU(),
                nn.Dropout(self.drop_rate),
                nn.Linear(self.flat_dim, self.flat_dim),
            )

    def _assert_agent_inputs(self, batch_size, agent_cond_embed, agent_gate_prior):
        assert agent_cond_embed is not None, 'CPTransformer expected agent_cond_embed when args.use_agent is true.'
        assert agent_cond_embed.dim() == 2 and agent_cond_embed.shape == (batch_size, self.d_llm), (
            f'CPTransformer expected agent_cond_embed shape [{batch_size}, {self.d_llm}], '
            f'but got {tuple(agent_cond_embed.shape)}.'
        )
        assert agent_gate_prior is not None, 'CPTransformer expected agent_gate_prior when args.use_agent is true.'
        assert agent_gate_prior.dim() == 2 and agent_gate_prior.shape == (batch_size, self.num_total_experts), (
            f'CPTransformer expected agent_gate_prior shape [{batch_size}, {self.num_total_experts}], '
            f'but got {tuple(agent_gate_prior.shape)}.'
        )

    def _prepare_agent_confidence(self, batch_size, agent_confidence, device, dtype):
        if agent_confidence is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if agent_confidence.dim() == 1:
            agent_confidence = agent_confidence.unsqueeze(-1)
        else:
            assert agent_confidence.dim() == 2 and agent_confidence.shape[1] == 1, (
                f'CPTransformer expected agent_confidence shape [{batch_size}] or [{batch_size}, 1], '
                f'but got {tuple(agent_confidence.shape)}.'
            )
        assert agent_confidence.shape[0] == batch_size, (
            f'CPTransformer expected agent_confidence batch size {batch_size}, '
            f'but got {agent_confidence.shape[0]}.'
        )
        return agent_confidence.to(device=device, dtype=dtype)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False, DKP_embeddings=None, cathode_masks=None,
        temperature_masks=None, format_masks=None, anode_masks=None, combined_masks=None, ion_type_masks=None, use_view_experts=False,
        agent_cond_embed=None, agent_gate_prior=None, agent_confidence=None):
        '''
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        '''
        batch_size = cycle_curve_data.shape[0]
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros

        cycle_curve_data = self.intra_flatten(cycle_curve_data) # [B, early_cycle, fixed_len * num_var]
        cycle_curve_data = self.intra_embed(cycle_curve_data)
        for i in range(self.e_layers):
            cycle_curve_data = self.intra_MLP[i](cycle_curve_data) # [B, early_cycle, d_model]

        cycle_curve_data = self.pe(cycle_curve_data) + cycle_curve_data
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L]
        curve_attn_mask = torch.repeat_interleave(curve_attn_mask, curve_attn_mask.shape[-1], dim=1) # [B, L, L]
        curve_attn_mask = curve_attn_mask.unsqueeze(1) # [B, 1, L, L]
        curve_attn_mask = curve_attn_mask==0 # set True to mask
        output, attns = self.inter_TransformerEncoder(cycle_curve_data, attn_mask=curve_attn_mask)

        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, L * d_model)
        if self.use_agent:
            self._assert_agent_inputs(batch_size, agent_cond_embed, agent_gate_prior)
            agent_confidence = self._prepare_agent_confidence(
                batch_size, agent_confidence, output.device, output.dtype
            )
            agent_input = torch.cat(
                [agent_cond_embed.to(device=output.device, dtype=output.dtype), agent_confidence],
                dim=-1,
            )
            agent_vec = self.agent_projector(agent_input)
            output = self.agent_fusion_mlp(torch.cat([output, agent_vec], dim=-1))
        preds = self.projection(output)  # (batch_size, num_classes)
        return preds, None, None, None, None, None, 0.0, 0.0
