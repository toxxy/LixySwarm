"""
Lixy-0.1 — Bio-Inspired Agent Architecture
Capa Hormiga: AgentBase

Cada agente es un transformer pequeño (125M params) con dos modificaciones clave:
1. Feromon Vector: recibe señales de 256-dim de los agentes vecinos
2. Identity Vector: embedding fijo (no entrenable) que define la "especialidad" del agente

Basado en nanoGPT (Karpathy) con extensiones.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    # Arquitectura base (equivalente a GPT2-small)
    block_size: int = 1024       # context length
    vocab_size: int = 50304      # GPT2 vocab padded to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # Extensiones Lixy
    feromon_dim: int = 256       # dimensión del vector de feromona
    identity_dim: int = 64       # dimensión del embedding de identidad
    agent_id: int = 0            # 0=léxico, 1=semántico, 2=generación, etc.
    n_agents: int = 3            # total de agentes en el enjambre


class LayerNorm(nn.Module):
    """LayerNorm con bias opcional."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash Attention (PyTorch 2.x)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class FeromonGate(nn.Module):
    """
    🐜 Mecanismo de Feromona Digital
    
    Integra señales del enjambre en el procesamiento del agente.
    Análogo: una hormiga ajusta su comportamiento según la concentración
    de feromonas locales antes de decidir su próxima acción.
    
    Input: embedding del token [B, T, n_embd] + feromona [B, feromon_dim]
    Output: embedding modulado [B, T, n_embd]
    """
    def __init__(self, config):
        super().__init__()
        self.feromon_proj = nn.Linear(config.feromon_dim, config.n_embd, bias=False)
        self.gate = nn.Linear(config.n_embd * 2, config.n_embd, bias=config.bias)
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, feromon: Optional[torch.Tensor] = None):
        if feromon is None:
            return x
        # feromona: [B, feromon_dim] → [B, 1, n_embd] → broadcast
        f = self.feromon_proj(feromon).unsqueeze(1)  # [B, 1, n_embd]
        f = f.expand_as(x)  # [B, T, n_embd]
        # Gate: cuánto dejar pasar de la señal de feromona
        gate_input = torch.cat([x, f], dim=-1)  # [B, T, n_embd*2]
        gate = torch.sigmoid(self.gate(gate_input))
        return self.norm(x + gate * f)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AgentBase(nn.Module):
    """
    🐜 Agente Base del Enjambre Lixy-0.1
    
    Un transformer pequeño (125M params) con soporte para:
    - Vectores de feromona (señales del enjambre)
    - Embedding de identidad (silbido único del Delfín)
    - Salida de feromona (qué señal emite este agente hacia el enjambre)
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config

        # ─── Embedding de identidad (no entrenable — el "silbido") ───
        # Cada agente tiene un ID fijo que colorea todas sus representaciones
        identity = torch.randn(config.identity_dim) * 0.02
        self.register_buffer('identity_vec', identity)
        self.identity_proj = nn.Linear(config.identity_dim, config.n_embd, bias=False)

        # ─── Transformer base ───
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # ─── Feromona Gate (antes del primer layer) ───
        self.feromon_gate = FeromonGate(config)

        # ─── Cabeza de lenguaje ───
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # ─── Cabeza de feromona de salida ───
        # Qué señal emite este agente para los demás
        self.feromon_out = nn.Linear(config.n_embd, config.feromon_dim, bias=False)

        # Inicialización
        self.apply(self._init_weights)
        # Escalar proyecciones residuales
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"AgentBase [{config.agent_id}] inicializado: {n_params/1e6:.1f}M params")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,                      # [B, T] token ids
        targets: Optional[torch.Tensor] = None,  # [B, T] para calcular loss
        feromon_in: Optional[torch.Tensor] = None  # [B, feromon_dim] señal del enjambre
    ):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Secuencia de largo {T} excede block_size {self.config.block_size}"

        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # [T]

        # ─── Embeddings ───
        tok_emb = self.transformer.wte(idx)   # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos)   # [T, n_embd]

        # Sumar identidad del agente (el "silbido" 🐬)
        identity = self.identity_proj(self.identity_vec)  # [n_embd]
        x = self.transformer.drop(tok_emb + pos_emb + identity)

        # ─── Gate de feromona (entrada del enjambre) ───
        x = self.feromon_gate(x, feromon_in)

        # ─── Transformer layers ───
        if getattr(self, 'use_gradient_checkpointing', False) and self.training:
            from torch.utils.checkpoint import checkpoint as ckpt_fn
            for block in self.transformer.h:
                x = ckpt_fn(block, x, use_reentrant=False)
        else:
            for block in self.transformer.h:
                x = block(x)
        x = self.transformer.ln_f(x)

        # ─── Salida ───
        # Feromona de salida: qué señal emite este agente
        # Usamos el promedio del último estado oculto como "resumen"
        feromon_out = self.feromon_out(x.mean(dim=1))  # [B, feromon_dim]
        feromon_out = torch.tanh(feromon_out)  # normalizar en [-1, 1]

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # Solo calcular logits del último token (inferencia)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, feromon_out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, feromon_in=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond, feromon_in=feromon_in)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Test rápido
    config = AgentConfig(agent_id=0)
    model = AgentBase(config)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Simular un batch
    device = next(model.parameters()).device
    x = torch.randint(0, config.vocab_size, (2, 64), device=device)
    feromon = torch.randn(2, config.feromon_dim, device=device)

    logits, loss, feromon_out = model(x, targets=x, feromon_in=feromon)
    print(f"logits: {logits.shape}")
    print(f"feromon_out: {feromon_out.shape}")
    print(f"loss: {loss.item():.4f} (esperado ~ln({config.vocab_size})={math.log(config.vocab_size):.2f})")
    print("✓ AgentBase funcionando")
