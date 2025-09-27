import torch
import torch.nn as nn
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM
import torch.nn.functional as F

def build_custom_attention_mask(seq_len, user_seq_len, device='cpu'):
    i = torch.arange(seq_len, device=device).view(-1, 1)  # shape: (L, 1)
    j = torch.arange(seq_len, device=device).view(1, -1)  # shape: (1, L)
    causal_mask = (i >= user_seq_len) & (i < j)
    user_to_assistant_mask = (i < user_seq_len) & (j >= user_seq_len)
    mask = causal_mask | user_to_assistant_mask

    return mask

class AttentionLayer(nn.Module):
    def __init__(self, input_dim=3584, hidden_dim=128):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.register_buffer("causal_mask", torch.triu(torch.ones(1, 8192, 8192), diagonal=1).bool())

    def forward(self, x, user_seq_len=None):
        """
        x: shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        Q = self.query(x)  # (batch, seq_len, hidden_dim)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / ((K.size(-1) + 1e-6) ** 0.5)  # (batch, seq_len, seq_len)

        if user_seq_len is None:
            mask = self.causal_mask[:, :seq_len, :seq_len]  # (1, 1, seq_len, seq_len)
        else:
            mask = build_custom_attention_mask(seq_len, user_seq_len, x.device)
        attention_scores = attention_scores.masked_fill(mask, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, V)

        return context_vector, attention_weights


class CfcCell(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(CfcCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wz = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Uz = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bz = nn.Parameter(torch.zeros(hidden_dim))

        self.Wr = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Ur = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.br = nn.Parameter(torch.zeros(hidden_dim))

        self.Wh = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Uh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]:
            torch.nn.init.orthogonal_(weight)

    def forward(self, x, h_prev, dt):
        z = torch.sigmoid(torch.matmul(x, self.Wz) + torch.matmul(h_prev, self.Uz) + self.bz)
        r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_prev, self.Ur) + self.br)
        h_hat = torch.tanh(torch.matmul(x, self.Wh) + torch.matmul(r * h_prev, self.Uh) + self.bh)
        h_new = (1 - z) * h_prev + z * h_hat

        return h_new + dt * (h_new - h_prev)


class StreamingSafetyHead(nn.Module):
    def __init__(self,
             input_dim=4096,
             proj_dim=512,
             mem_dim=512,
             num_labels=1,
             use_dt=False,
             dt=1.0,
             dropout=0.1,
         ):
        super().__init__()
        self.attention = AttentionLayer(input_dim, proj_dim)

        d_in = proj_dim
        self.cfc = CfcCell(d_in, mem_dim)
        self.mem_head = nn.Linear(mem_dim, num_labels)

        self.prefix_to_h = nn.Sequential(
            nn.Linear(d_in, mem_dim),
            nn.Tanh()
        )
        self.prefix_scorer = nn.Linear(d_in, 1, bias=False)

        self._h = None  # (B, mem_dim)

    def reset_state(self, batch_size=None, device=None, dtype=None):
        self._h = None
        if batch_size is not None:
            if device is None:
                device = next(self.parameters()).device
            if dtype is None:
                dtype = next(self.parameters()).dtype
            self._h = torch.zeros(batch_size, self.cfc.hidden_dim, device=device, dtype=dtype)

    def _ensure_state(self, x):
        if self._h is None or self._h.size(0) != x.size(0) or self._h.device != x.device or self._h.dtype != x.dtype:
            self.reset_state(batch_size=x.size(0), device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def init_with_prefix(self, assist_len, user_hidden, stride=64):

        device, dtype = user_hidden.device, user_hidden.dtype
        B, U, D = user_hidden.shape
        self.reset_state(batch_size=B, device=device, dtype=dtype)

        times = torch.linspace(0, 1, steps=assist_len).to(device)
        self.dt = torch.zeros_like(times)
        self.dt[1:] = times[1:] - times[:-1]

        x = user_hidden

        scores = self.prefix_scorer(x).squeeze(-1)     # (B, U)
        weights = torch.softmax(scores, dim=1)         # (B, U)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, d_in)
        self._h = self.prefix_to_h(pooled)
        return


    def step(self, x_t, t):

        self._ensure_state(x_t)
        self._h = self.cfc(x_t, self._h, self.dt[t])
        mem_logits = self.mem_head(self._h)  # (B, C)

        return mem_logits

    def forward(self, x, assistant_start, is_multi=False):

        feat =self.attention(x)[0]

        seq_total = feat.shape[1]
        run_len = seq_total - assistant_start
        prefix = feat[:, :assistant_start, :]
        self.init_with_prefix(run_len, prefix)

        logits = torch.cat([self.step(feat[:, assistant_start+t, :], t).unsqueeze(1) for t in range(run_len)], dim=1)

        return logits



