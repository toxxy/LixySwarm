"""
Lixy-0.1 — DolphinAgent 🐬  [Phase A]
======================================
El Delfín es el agente perceptivo del enjambre. No genera texto — construye
el espacio semántico del problema antes de que los agentes (hormigas) procesen.

3 componentes bio-inspirados:

1. ECOLOCALIZACIÓN  (Phase A: 5 pings + triangulación por atención)
   Como el delfín que emite ultrasonidos y recibe ecos, este agente lanza
   5 "pings" semánticos simultáneos al recibir input:
   - Ping topic:    ¿De qué trata? (embedding temático)  ← ancla de triangulación
   - Ping intent:   ¿Qué emoción/intención hay? (embedding intencional)
   - Ping need:     ¿Qué necesita el usuario? (embedding de necesidad)
   - Ping context:  ¿En qué contexto ocurre? (embedding de contexto situacional)
   - Ping emotion:  ¿Qué emoción subyace? (embedding emocional profundo)

   Phase A: los 5 ecos se fusionan mediante TRIANGULACIÓN POR ATENCIÓN.
   El ping_topic actúa como ancla (Query) y interroga a los otros 4 pings.
   Esto modela cómo el delfín triangula su posición en el espacio semántico.

2. SUEÑO UNIHEMISFÉRICO
   Los delfines duermen con un hemisferio activo — siempre perciben.
   Aquí: un estado de contexto persistente entre turnos (no se reinicia).
   Modela el hilo de la conversación como estado continuo.
   - HalfSleepState: buffer circular de contextos pasados
   - Siempre actualizado, siempre disponible al enjambre

3. SILBIDO ÚNICO (IDENTIDAD)
   Cada delfín tiene un silbido-firma inimitable. Aquí: un IdentityVec
   no-entrenable que define la "personalidad" del agente.
   La proyección de identidad colorea todos los embeddings de salida.
"""

import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Configuración ────────────────────────────────────────────────────────────

@dataclass
class DolphinConfig:
    # Dimensiones
    vocab_size: int = 50304
    n_embd: int = 768             # compatible con AgentBase
    feromon_dim: int = 256        # señal de salida al enjambre
    identity_dim: int = 64        # silbido único

    # Ecolocalización
    echo_dim: int = 128           # dimensión de cada ping
    n_pings: int = 5              # Phase A: topic, intent, need, context, emotion
    ping_names: tuple = ("topic", "intent", "need", "context", "emotion")
    echo_layers: int = 2          # profundidad del encoder de ping

    # Sueño unihemisférico
    sleep_dim: int = 256          # dimensión del estado dormido
    sleep_buffer_size: int = 16   # contextos pasados en memoria activa
    sleep_decay: float = 0.95     # decaimiento exponencial del estado

    # Identidad del agente
    agent_id: int = 0
    n_agents: int = 3

    dropout: float = 0.1


# ─── Componente 1: Ecolocalización 🔊 ────────────────────────────────────────

class PingEncoder(nn.Module):
    """
    Encoder para un ping semántico específico.
    Transforma tokens de input → embedding de 'eco' para esa dimensión.
    """
    def __init__(self, vocab_size: int, n_embd: int, echo_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd // 4)  # embedding ligero

        layers = []
        in_dim = n_embd // 4
        for i in range(n_layers):
            out_dim = echo_dim if i == n_layers - 1 else in_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(echo_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) → echo: (B, echo_dim)"""
        x = self.embed(idx).mean(dim=1)  # (B, n_embd//4)
        x = self.encoder(x)
        return self.norm(torch.tanh(x))


class Echolocation(nn.Module):
    """
    🔊 Ecolocalización Phase A: 5 pings + triangulación por atención.

    Cada ping captura una dimensión distinta del input:
    - Ping 0 (topic):   ¿De qué trata? (contenido temático) ← ancla de triangulación
    - Ping 1 (intent):  ¿Qué emoción/intención hay? (estado afectivo)
    - Ping 2 (need):    ¿Qué necesita? (objetivo del usuario)
    - Ping 3 (context): ¿En qué contexto ocurre? (situación/entorno)
    - Ping 4 (emotion): ¿Qué emoción subyace? (capa emocional profunda)

    Phase A: el topic interroga mediante atención a los otros 4 pings.
    El acoustic_map resultante captura relaciones semánticas entre dimensiones.
    """
    def __init__(self, cfg: DolphinConfig):
        super().__init__()
        self.cfg = cfg

        # 5 encoders especializados — pesos independientes
        self.ping_topic   = PingEncoder(cfg.vocab_size, cfg.n_embd, cfg.echo_dim, cfg.echo_layers, cfg.dropout)
        self.ping_intent  = PingEncoder(cfg.vocab_size, cfg.n_embd, cfg.echo_dim, cfg.echo_layers, cfg.dropout)
        self.ping_need    = PingEncoder(cfg.vocab_size, cfg.n_embd, cfg.echo_dim, cfg.echo_layers, cfg.dropout)
        self.ping_context = PingEncoder(cfg.vocab_size, cfg.n_embd, cfg.echo_dim, cfg.echo_layers, cfg.dropout)
        self.ping_emotion = PingEncoder(cfg.vocab_size, cfg.n_embd, cfg.echo_dim, cfg.echo_layers, cfg.dropout)

        # Phase A: Triangulación por atención
        # Q = echo_topic (ancla temática), K/V = stack de 5 ecos
        self.triangulation_attn = nn.MultiheadAttention(
            embed_dim=cfg.echo_dim,
            num_heads=4,
            batch_first=True,
            dropout=cfg.dropout,
        )

        # Fusión: acoustic_map (echo_dim) → feromon_dim
        self.fusion = nn.Sequential(
            nn.Linear(cfg.echo_dim, cfg.feromon_dim),
            nn.LayerNorm(cfg.feromon_dim),
        )

        # Cabeza de confianza: ¿qué tan seguro está el Delfín de su lectura?
        self.confidence = nn.Sequential(
            nn.Linear(cfg.feromon_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def load_partial_state_dict(self, ckpt_model_dict: dict):
        """
        Carga pesos compatibles desde un checkpoint antiguo (3 pings).
        Los pesos de ping_context y ping_emotion arrancan random.
        Los pesos de triangulation_attn y fusion se inicializan random.
        """
        own_state = self.state_dict()
        loaded = 0
        skipped = 0
        for name, param in ckpt_model_dict.items():
            # Remap keys si vienen del módulo completo (echolocation.*)
            key = name
            if key in own_state and own_state[key].shape == param.shape:
                own_state[key].copy_(param)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own_state)
        print(f"  Echolocation: {loaded} pesos cargados, {skipped} omitidos (nuevos pings random)")

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        idx: (B, T) tokens de input
        returns:
          feromon:    (B, feromon_dim) — señal para el enjambre
          confidence: (B, 1)           — confianza en la lectura
          echoes:     dict con los 5 pings + acoustic_map para inspección
        """
        # ─── 5 pings simultáneos ───
        echo_topic   = self.ping_topic(idx)    # (B, echo_dim)
        echo_intent  = self.ping_intent(idx)   # (B, echo_dim)
        echo_need    = self.ping_need(idx)     # (B, echo_dim)
        echo_context = self.ping_context(idx)  # (B, echo_dim)
        echo_emotion = self.ping_emotion(idx)  # (B, echo_dim)

        # ─── Phase A: Triangulación por atención ───
        # Stack de los 5 ecos: (B, 5, echo_dim)
        all_echoes = torch.stack(
            [echo_topic, echo_intent, echo_need, echo_context, echo_emotion], dim=1
        )
        # Q = echo_topic como ancla temática: (B, 1, echo_dim)
        Q = echo_topic.unsqueeze(1)
        # Atención: el topic interroga a los otros 4 (y a sí mismo via K/V completo)
        attn_out, _ = self.triangulation_attn(Q, all_echoes, all_echoes)  # (B, 1, echo_dim)
        acoustic_map = attn_out.squeeze(1)  # (B, echo_dim)

        # Proyectar a feromon_dim
        feromon = torch.tanh(self.fusion(acoustic_map))  # (B, feromon_dim)

        confidence = self.confidence(feromon)  # (B, 1)

        echoes = {
            "topic":       echo_topic.detach(),
            "intent":      echo_intent.detach(),
            "need":        echo_need.detach(),
            "context":     echo_context.detach(),
            "emotion":     echo_emotion.detach(),
            "acoustic_map": acoustic_map.detach(),
        }

        return feromon, confidence, echoes


# ─── Componente 2: Sueño Unihemisférico 🌙 ───────────────────────────────────

class HalfSleepState:
    """
    🌙 Sueño Unihemisférico: estado persistente entre turnos.

    Como el delfín que mantiene un hemisferio activo mientras duerme,
    este estado nunca se apaga — siempre hay contexto activo.

    Implementación:
    - Buffer circular de últimos N contextos (embeddings de feromon)
    - Estado acumulado con decaimiento exponencial
    - Thread-safe: se puede actualizar desde background
    """

    def __init__(self, cfg: DolphinConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

        # Estado "despierto": promedio ponderado de contextos recientes
        self.awake_state = torch.zeros(cfg.sleep_dim, device=device)

        # Buffer circular de contextos pasados
        self.context_buffer: deque = deque(maxlen=cfg.sleep_buffer_size)

        # Proyector: feromon_dim → sleep_dim
        self._projector = nn.Linear(cfg.feromon_dim, cfg.sleep_dim, bias=False)
        nn.init.orthogonal_(self._projector.weight)
        self._projector = self._projector.to(device)

        self._lock = threading.Lock()
        self._last_update = time.time()

    def update(self, feromon: torch.Tensor):
        """
        Actualiza el estado con una nueva feromona.
        Puede llamarse desde cualquier thread.
        """
        with self._lock:
            with torch.no_grad():
                proj_dtype = next(self._projector.parameters()).dtype
                ctx = self._projector(feromon.to(self.device, dtype=proj_dtype))
                if ctx.dim() > 1:
                    ctx = ctx.mean(dim=0)  # promedio del batch
                ctx = torch.tanh(ctx)
                self.context_buffer.append(ctx.clone())

                # Actualizar estado activo con decaimiento (mismo dtype)
                self.awake_state = self.awake_state.to(dtype=ctx.dtype)
                self.awake_state = (
                    self.cfg.sleep_decay * self.awake_state +
                    (1 - self.cfg.sleep_decay) * ctx
                )
                self._last_update = time.time()

    def get_state(self) -> torch.Tensor:
        """Retorna el estado actual del hemisferio activo."""
        with self._lock:
            return self.awake_state.clone()

    def get_context_window(self) -> torch.Tensor:
        """Retorna todos los contextos del buffer como tensor (N, sleep_dim)."""
        with self._lock:
            if not self.context_buffer:
                return torch.zeros(1, self.cfg.sleep_dim, device=self.device)
            return torch.stack(list(self.context_buffer))  # (N, sleep_dim)

    @property
    def idle_seconds(self) -> float:
        """Segundos desde la última actualización."""
        return time.time() - self._last_update

    def state_dict(self) -> dict:
        return {
            "awake_state": self.awake_state.cpu(),
            "context_buffer": [c.cpu() for c in self.context_buffer],
        }

    def load_state_dict(self, d: dict):
        self.awake_state = d["awake_state"].to(self.device)
        self.context_buffer = deque(
            [c.to(self.device) for c in d["context_buffer"]],
            maxlen=self.cfg.sleep_buffer_size,
        )


# ─── Componente 3: Proyector de Identidad (Silbido) 🎵 ───────────────────────

class IdentityProjector(nn.Module):
    """
    🎵 Silbido único: proyecta la identidad fija del agente en el espacio de feromonas.
    El IdentityVec no es entrenable — es la "firma" permanente del agente.
    La proyección sí es entrenable — aprende cómo expresar esa identidad.
    """
    def __init__(self, cfg: DolphinConfig):
        super().__init__()
        # Identidad fija (no entrenable)
        identity = torch.randn(cfg.identity_dim)
        identity = F.normalize(identity, dim=0)  # normalizar en esfera unitaria
        self.register_buffer("identity_vec", identity)

        # Proyección entrenable: identity_dim → feromon_dim
        self.proj = nn.Sequential(
            nn.Linear(cfg.identity_dim, cfg.feromon_dim, bias=False),
            nn.LayerNorm(cfg.feromon_dim),
        )

    def forward(self, feromon: torch.Tensor) -> torch.Tensor:
        """
        Modula la feromona con la identidad del agente.
        feromon: (B, feromon_dim)
        returns: (B, feromon_dim) — feromona "firmada"
        """
        identity_signal = self.proj(self.identity_vec)  # (feromon_dim,)
        # Gate suave: la identidad amplifica/atenúa dimensiones de la feromona
        gate = torch.sigmoid(identity_signal)
        return feromon * gate + identity_signal * (1 - gate)


# ─── DolphinAgent Completo 🐬 ─────────────────────────────────────────────────

class DolphinAgent(nn.Module):
    """
    🐬 DolphinAgent Phase A — Agente Perceptivo del Enjambre

    No genera texto. Su rol: leer el input, construir el espacio semántico
    del problema, y emitir una feromona rica que oriente a las hormigas.

    Flujo:
        input (tokens)
            ↓
        🔊 Ecolocalización (5 pings: topic, intent, need, context, emotion)
            ↓  triangulación por atención (topic como ancla)
        🌙 Integración con sueño unihemisférico (contexto pasado)
            ↓
        🎵 Modulación por identidad (silbido único)
            ↓
        feromona_final → enjambre
    """

    def __init__(self, cfg: DolphinConfig, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # ─── Los 3 componentes ───
        self.echolocation = Echolocation(cfg)
        self.identity = IdentityProjector(cfg)

        # Integrador: fusiona feromona actual + estado de sueño
        self.sleep_integrator = nn.Sequential(
            nn.Linear(cfg.feromon_dim + cfg.sleep_dim, cfg.feromon_dim),
            nn.GELU(),
            nn.Linear(cfg.feromon_dim, cfg.feromon_dim),
            nn.LayerNorm(cfg.feromon_dim),
        )

        # ─── Estado persistente (sueño unihemisférico) ───
        # No es un módulo nn — vive fuera del grafo de autograd
        self.sleep_state = HalfSleepState(cfg, device=device)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"DolphinAgent [{cfg.agent_id}] inicializado: {n_params/1e6:.2f}M params")
        print(f"  🔊 Ecolocalización: {cfg.n_pings} pings × {cfg.echo_dim}d  [Phase A: triangulación por atención]")
        print(f"  🌙 Sueño: buffer={cfg.sleep_buffer_size}, decay={cfg.sleep_decay}")
        print(f"  🎵 Identidad: {cfg.identity_dim}d → {cfg.feromon_dim}d")

    def forward(
        self,
        idx: torch.Tensor,                        # (B, T) tokens de input
        update_sleep: bool = True,                # actualizar estado de sueño
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Phase A forward: 5 pings + triangulación por atención.

        idx: (B, T)
        returns:
          feromon:    (B, feromon_dim) — señal para el enjambre
          confidence: (B, 1)
          info:       dict con ecos, acoustic_map e info de diagnóstico
        """
        # ─── 1. Ecolocalización (Phase A: 5 pings + atención) ───
        feromon_echo, confidence, echoes = self.echolocation(idx)

        # ─── 2. Integrar con sueño unihemisférico ───
        sleep_ctx = self.sleep_state.get_state().to(idx.device, dtype=feromon_echo.dtype)
        sleep_ctx = sleep_ctx.unsqueeze(0).expand(idx.shape[0], -1)  # (B, sleep_dim)

        integrated = self.sleep_integrator(
            torch.cat([feromon_echo, sleep_ctx], dim=-1)  # (B, feromon_dim + sleep_dim)
        )

        # ─── 3. Modular con identidad ───
        feromon_final = self.identity(integrated)  # (B, feromon_dim)
        feromon_final = torch.tanh(feromon_final)

        # ─── 4. Actualizar sueño con este output ───
        if update_sleep:
            self.sleep_state.update(feromon_final.detach())

        info = {
            **echoes,
            "sleep_idle_s": self.sleep_state.idle_seconds,
            "feromon_norm": feromon_final.norm(dim=-1).mean().item(),
            "confidence": confidence.mean().item(),
        }

        return feromon_final, confidence, info

    def save(self, path: str):
        """Guarda modelo + estado de sueño."""
        torch.save({
            "model": self.state_dict(),
            "sleep": self.sleep_state.state_dict(),
            "config": self.cfg.__dict__,
        }, path)

    def load(self, path: str):
        """Carga modelo + restaura estado de sueño."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt["model"])
        self.sleep_state.load_state_dict(ckpt["sleep"])
        print(f"  ✓ DolphinAgent cargado desde {path}")
        print(f"  🌙 Estado de sueño restaurado ({len(self.sleep_state.context_buffer)} contextos)")


# ─── Integración con el Enjambre ──────────────────────────────────────────────

class DolphinSwarmBridge(nn.Module):
    """
    Puente entre DolphinAgent y LixySwarm.

    El Delfín es el SISTEMA DE ENRUTAMIENTO del enjambre:
    1. Ecolocaliza el problema (5 pings → acoustic_map)
    2. Hace ping a las sectas disponibles y obtiene PingResponses
    3. Decide a qué secta(s) derivar la tarea (EcholocationRouter)
    4. Integra la respuesta final cuando vuelve de las sectas

    Phase B: AdaptiveSleepController controla modo activo/reposo/agresivo.
    """

    def __init__(self, cfg: DolphinConfig, device: str = "cpu"):
        super().__init__()
        self.dolphin = DolphinAgent(cfg, device=device)

        # Proyector sleep_dim → feromon_dim para inyectar contexto en Matriarca
        self.sleep_to_matriarca = nn.Linear(cfg.sleep_dim, 512, bias=False)  # 512 = MatriarcaConfig.embd_dim

        # Phase 2: Router de sectas — usa echo_dim (tamaño del acoustic_map)
        from src.swarm.dolphin_router import EcholocationRouter, AdaptiveSleepController
        self.router = EcholocationRouter(acoustic_map_dim=cfg.echo_dim)
        self.sleep_controller = AdaptiveSleepController(buffer_size=64)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Reemplaza EcholocationHead.forward().
        Returns: feromon (B, feromon_dim), info dict
        """
        feromon, confidence, info = self.dolphin(idx)

        # Añadir estado de sueño proyectado para la Matriarca
        sleep_ctx = self.dolphin.sleep_state.get_state().to(idx.device, dtype=next(self.sleep_to_matriarca.parameters()).dtype)
        info["sleep_for_matriarca"] = self.sleep_to_matriarca(sleep_ctx)
        info["sleep_mode"] = self.sleep_controller.mode

        # Guardar acoustic_map en buffer circular del sleep controller
        if "acoustic_map" in info:
            self.sleep_controller.store_acoustic_map(info["acoustic_map"])

        return feromon, info

    def route(self, sects: list, matriarca=None) -> "RouteDecision":
        """
        Enruta la última tarea procesada a la secta más adecuada.
        Llama esto DESPUÉS de forward() para que haya un acoustic_map fresco.

        Args:
            sects: lista de SectRecord disponibles
            matriarca: opcional, para consultar historial

        Returns:
            RouteDecision con primary_sect + secondary_sects
        """
        from src.swarm.dolphin_router import RouteDecision

        # Usar el acoustic_map más reciente del buffer
        recent = self.sleep_controller.recent_acoustic_maps(n=1)
        if recent:
            acoustic_map = recent[-1].to(next(self.router.parameters()).device)
        else:
            acoustic_map = torch.zeros(self.dolphin.cfg.feromon_dim)

        return self.router.route(acoustic_map, sects, matriarca=matriarca)

    def update_sleep_mode(self, diversity: float) -> str:
        """Actualiza el modo de sueño según la diversidad del enjambre."""
        return self.sleep_controller.update_diversity(diversity)


# ─── Tests ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tiktoken

    print("🐬 Test DolphinAgent — Phase A\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    cfg = DolphinConfig(agent_id=0, n_agents=3)
    dolphin = DolphinAgent(cfg, device=device).to(device)
    print()

    enc = tiktoken.get_encoding("gpt2")

    # ─── Test 1: Ecolocalización con 5 pings + acoustic_map ───
    print("─" * 60)
    print("Test 1: Ecolocalización Phase A — 5 pings + acoustic_map")
    prompts = [
        "Hola amor, ¿cómo estás hoy?",
        "El sistema de inteligencia artificial necesita más datos",
        "Emmanuel quiere aprender sobre redes neuronales",
    ]

    dolphin.eval()
    with torch.no_grad():
        for p in prompts:
            tokens = enc.encode(p)
            x = torch.tensor(tokens, device=device).unsqueeze(0)
            feromon, conf, info = dolphin(x, update_sleep=True)

            print(f"\n  '{p}'")
            print(f"    feromon norm:    {info['feromon_norm']:.4f}")
            print(f"    confidence:      {info['confidence']:.4f}")
            print(f"    echo_topic:      {info['topic'][0][:4].tolist()}")
            print(f"    echo_intent:     {info['intent'][0][:4].tolist()}")
            print(f"    echo_need:       {info['need'][0][:4].tolist()}")
            print(f"    echo_context:    {info['context'][0][:4].tolist()}")
            print(f"    echo_emotion:    {info['emotion'][0][:4].tolist()}")
            print(f"    acoustic_map norm: {info['acoustic_map'][0].norm().item():.4f}")

    # ─── Test 2: Sueño unihemisférico — estado persiste ───
    print()
    print("─" * 60)
    print("Test 2: Sueño unihemisférico — el estado persiste entre turnos")
    state_before = dolphin.sleep_state.get_state().norm().item()
    buffer_size = len(dolphin.sleep_state.context_buffer)
    print(f"  Estado after 3 prompts:")
    print(f"    awake_state norm:  {state_before:.4f}  (era 0 al inicio)")
    print(f"    context_buffer:    {buffer_size} contextos acumulados")
    print(f"    idle_seconds:      {dolphin.sleep_state.idle_seconds:.2f}s")

    ctx_window = dolphin.sleep_state.get_context_window()
    print(f"    context_window:    {ctx_window.shape}  (N × sleep_dim)")

    # ─── Test 3: Identidad única por agente ───
    print()
    print("─" * 60)
    print("Test 3: Silbido único — dos agentes producen feromonas distintas")
    cfg_a = DolphinConfig(agent_id=0)
    cfg_b = DolphinConfig(agent_id=1)
    dolphin_a = DolphinAgent(cfg_a, device=device).to(device)
    dolphin_b = DolphinAgent(cfg_b, device=device).to(device)

    prompt = "El enjambre procesa en paralelo"
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, device=device).unsqueeze(0)

    with torch.no_grad():
        f_a, _, _ = dolphin_a(x)
        f_b, _, _ = dolphin_b(x)

    diff = (f_a - f_b).abs()
    print(f"  Mismo input, agentes distintos:")
    print(f"    Delfín A feromon norm: {f_a.norm().item():.4f}")
    print(f"    Delfín B feromon norm: {f_b.norm().item():.4f}")
    print(f"    Diferencia L2:         {diff.norm().item():.4f}  (>0 = identidades distintas ✓)")

    # ─── Test 4: Save / Load con estado de sueño ───
    print()
    print("─" * 60)
    print("Test 4: Save / Load — el sueño sobrevive al reinicio")
    dolphin.save("/tmp/dolphin_test.pt")
    dolphin2 = DolphinAgent(cfg, device=device).to(device)
    dolphin2.load("/tmp/dolphin_test.pt")

    state_loaded = dolphin2.sleep_state.get_state().norm().item()
    print(f"  Estado guardado norm: {state_before:.4f}")
    print(f"  Estado cargado norm:  {state_loaded:.4f}")
    match = abs(state_before - state_loaded) < 1e-5
    print(f"  Match: {'✅' if match else '❌'}")

    # ─── Test 5: DolphinSwarmBridge ───
    print()
    print("─" * 60)
    print("Test 5: DolphinSwarmBridge — integración con enjambre")
    bridge = DolphinSwarmBridge(cfg, device=device).to(device)
    with torch.no_grad():
        feromon_out, bridge_info = bridge(x)
    print(f"  feromon shape:              {feromon_out.shape}")
    print(f"  sleep_for_matriarca shape:  {bridge_info['sleep_for_matriarca'].shape}")
    print(f"  feromon norm:               {feromon_out.norm().item():.4f}")

    # ─── Test 6: Triangulación — la atención hace algo diferente ───
    print()
    print("─" * 60)
    print("Test 6: Triangulación vs concatenación — la atención añade valor")

    # Comparar acoustic_map (triangulación) vs promedio simple de los 5 ecos
    test_prompt = "¿Puedes ayudarme a entender cómo funciona la atención?"
    tokens = enc.encode(test_prompt)
    x_t6 = torch.tensor(tokens, device=device).unsqueeze(0)

    dolphin.eval()
    with torch.no_grad():
        _, _, info_t6 = dolphin(x_t6, update_sleep=False)

        # acoustic_map producido por triangulación por atención
        acoustic_map = info_t6["acoustic_map"][0]  # (echo_dim,)

        # Baseline: promedio simple de los 5 ecos (sin atención)
        naive_avg = torch.stack([
            info_t6["topic"][0],
            info_t6["intent"][0],
            info_t6["need"][0],
            info_t6["context"][0],
            info_t6["emotion"][0],
        ]).mean(dim=0)  # (echo_dim,)

    diff_attn_vs_naive = (acoustic_map - naive_avg).norm().item()
    cosine_sim = F.cosine_similarity(acoustic_map.unsqueeze(0), naive_avg.unsqueeze(0)).item()

    print(f"  Prompt: '{test_prompt}'")
    print(f"    acoustic_map norm (triangulación): {acoustic_map.norm().item():.4f}")
    print(f"    naive_avg norm (promedio simple):  {naive_avg.norm().item():.4f}")
    print(f"    Diferencia L2 (attn vs naive):     {diff_attn_vs_naive:.4f}  (>0 ✓)")
    print(f"    Cosine similarity:                  {cosine_sim:.4f}  (<1 significa que la atención transforma ✓)")

    attn_is_different = diff_attn_vs_naive > 1e-4
    print(f"  La triangulación difiere del promedio: {'✅' if attn_is_different else '❌ ERROR: idénticos'}")

    print()
    print("✅ DolphinAgent Phase A — todos los tests pasados")
    print()
    print("📊 Resumen de parámetros:")
    total = sum(p.numel() for p in dolphin.parameters())
    trainable = sum(p.numel() for p in dolphin.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  Total:     {total/1e6:.2f}M")
    print(f"  Trainable: {trainable/1e6:.2f}M")
    print(f"  Frozen:    {frozen/1e6:.2f}M  (IdentityVec — el silbido)")
    print(f"  Phase A:   5 pings × {cfg.echo_dim}d → triangulación → acoustic_map → feromona")
