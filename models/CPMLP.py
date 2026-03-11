import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

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

        self.inter_flatten = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.early_cycle_threshold*self.d_model, self.d_model))
        self.inter_MLP = nn.ModuleList([MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate) for _ in range(configs.d_layers)])
        self.head_output = nn.Linear(self.d_model, 1)
        if self.use_agent:
            self.agent_projector = nn.Sequential(
                nn.Linear(self.d_llm + 1, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.agent_fusion_mlp = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Dropout(self.drop_rate),
                nn.Linear(self.d_model, self.d_model),
            )

    def _assert_agent_inputs(self, batch_size, agent_cond_embed, agent_gate_prior):
        assert agent_cond_embed is not None, 'CPMLP expected agent_cond_embed when args.use_agent is true.'
        assert agent_cond_embed.dim() == 2 and agent_cond_embed.shape == (batch_size, self.d_llm), (
            f'CPMLP expected agent_cond_embed shape [{batch_size}, {self.d_llm}], '
            f'but got {tuple(agent_cond_embed.shape)}.'
        )
        assert agent_gate_prior is not None, 'CPMLP expected agent_gate_prior when args.use_agent is true.'
        assert agent_gate_prior.dim() == 2 and agent_gate_prior.shape == (batch_size, self.num_total_experts), (
            f'CPMLP expected agent_gate_prior shape [{batch_size}, {self.num_total_experts}], '
            f'but got {tuple(agent_gate_prior.shape)}.'
        )

    def _prepare_agent_confidence(self, batch_size, agent_confidence, device, dtype):
        if agent_confidence is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if agent_confidence.dim() == 1:
            agent_confidence = agent_confidence.unsqueeze(-1)
        else:
            assert agent_confidence.dim() == 2 and agent_confidence.shape[1] == 1, (
                f'CPMLP expected agent_confidence shape [{batch_size}] or [{batch_size}, 1], '
                f'but got {tuple(agent_confidence.shape)}.'
            )
        assert agent_confidence.shape[0] == batch_size, (
            f'CPMLP expected agent_confidence batch size {batch_size}, '
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

        cycle_curve_data = self.inter_flatten(cycle_curve_data) # [B, d_model]
        for i in range(self.d_layers):
            cycle_curve_data = self.inter_MLP[i](cycle_curve_data) # [B, d_model]

        if self.use_agent:
            self._assert_agent_inputs(batch_size, agent_cond_embed, agent_gate_prior)
            agent_confidence = self._prepare_agent_confidence(
                batch_size, agent_confidence, cycle_curve_data.device, cycle_curve_data.dtype
            )
            agent_input = torch.cat(
                [agent_cond_embed.to(device=cycle_curve_data.device, dtype=cycle_curve_data.dtype), agent_confidence],
                dim=-1,
            )
            agent_vec = self.agent_projector(agent_input)
            cycle_curve_data = self.agent_fusion_mlp(torch.cat([cycle_curve_data, agent_vec], dim=-1))

        preds = self.head_output(F.relu(cycle_curve_data))

        return preds, None, None, None, None, None, 0.0, 0.0
