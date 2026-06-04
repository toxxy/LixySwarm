"""
Lixy-0.1 — Orquestador del Enjambre v3
🐜🐘🐬 Hormiga + Elefante + Delfín — DELFÍN COMO AGENTE REAL

Cambios v3:
- DolphinAgent reemplaza EcholocationHead: el Delfín es ahora el
  primer agente real del enjambre (no solo cabeza de preprocesamiento).
- Sueño unihemisférico activo: el estado del Delfín persiste entre turnos.
- El estado de sueño del Delfín alimenta la Matriarca directamente.
- Tres capas bio-inspiradas completamente implementadas:
    🐬 Delfín → ecolocaliza el problema
    🐘 Matriarca → emite infrasónidos de orientación
    🐜 Hormigas ×3 → procesan con feromonas guiadas
"""

import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import json
import time
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.agent_base import AgentBase, AgentConfig
from src.matriarca.matriarca import Matriarca, MatriarcaConfig
from src.agents.dolphin_agent import DolphinAgent, DolphinSwarmBridge, DolphinConfig
from src.swarm.dolphin_pool import DolphinPool


# ─── Fitness de Agente 🐜 ─────────────────────────────────────────────────────────

@dataclass
class AgentFitness:
    """
    Métricas de fitness para un agente en un paso de training.
    El rol/label emerge del comportamiento, no se programa.
    """
    agent_id: int
    loss_contribution: float = 0.0   # cuánto redujo la loss vs promedio del enjambre
    feromon_divergence: float = 0.0  # qué tan diferente es su feromona de las otras
    confidence: float = 0.0          # cabeza de confianza del orquestador
    fitness: float = 0.0             # compuesto
    step: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "loss_contribution": round(self.loss_contribution, 4),
            "feromon_divergence": round(self.feromon_divergence, 4),
            "confidence": round(self.confidence, 4),
            "fitness": round(self.fitness, 4),
            "step": self.step,
        }


class SpecializationTracker:
    """
    Registra el historial de fitness y emergencia de especialización de cada agente.
    Los labels emergen del comportamiento, no se asignan manualmente.
    Guarda estado en checkpoints/ant_specialization.json.
    """
    SIMILARITY_THRESHOLD = 0.7   # cos_sim por encima de esto = demasiado parecidos

    def __init__(self, n_agents: int, checkpoint_dir: str = "checkpoints"):
        self.n_agents = n_agents
        self.path = Path(checkpoint_dir) / "ant_specialization.json"
        self.history: dict = {str(i): [] for i in range(n_agents)}
        self.current: dict = {str(i): AgentFitness(agent_id=i) for i in range(n_agents)}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.history = data.get("history", self.history)
                print(f"  🤝 Especialización cargada: {self.path.name}")
            except Exception:
                pass

    def save(self, step: int):
        # Calcular labels actuales
        current_labels = {k: self._infer_label(k) for k in self.current}
        data = {
            "step": step,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "labels": current_labels,   # labels actuales del run
            "history": self.history,
            "current": {k: v.to_dict() for k, v in self.current.items()},
            # Historial de labels por agente (para ver evolución entre runs)
            "label_history": {
                k: list(dict.fromkeys(  # deduplicar consecutivos
                    e.get("label", "?") for e in self.history[k][-100:]
                    if e.get("label") and e.get("label") != f"Agente-{k}"
                ))
                for k in self.current
            },
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def update(self, fitnesses: List[AgentFitness], step: int):
        """Actualiza estado con los fitness del step actual."""
        for af in fitnesses:
            k = str(af.agent_id)
            af.step = step
            self.current[k] = af
            entry = af.to_dict()
            # Guardar el label calculado en cada entrada del historial
            entry["label"] = self._infer_label(k)
            self.history[k].append(entry)
            # Mantener solo últimos 500 pasos
            if len(self.history[k]) > 500:
                self.history[k] = self.history[k][-500:]

    def _infer_label(self, agent_id_str: str) -> str:
        """
        Infiere un label basado en el comportamiento relativo al ENJAMBRE,
        no solo en el historial propio del agente.

        Requiere historial de todos los agentes para comparar entre ellos.
        """
        hist = self.history.get(agent_id_str, [])
        if len(hist) < 20:
            return f"Agente-{agent_id_str}"

        recent = hist[-30:]
        avg_div  = sum(h["feromon_divergence"] for h in recent) / len(recent)
        avg_conf = sum(h["confidence"] for h in recent) / len(recent)
        avg_fit  = sum(h["fitness"] for h in recent) / len(recent)

        # Comparar contra los OTROS agentes (no contra historial propio)
        # Esto da diferenciación real entre roles
        other_divs  = []
        other_confs = []
        other_fits  = []
        for other_id, other_hist in self.history.items():
            if other_id == agent_id_str or len(other_hist) < 20:
                continue
            other_recent = other_hist[-30:]
            other_divs.append(sum(h["feromon_divergence"] for h in other_recent) / len(other_recent))
            other_confs.append(sum(h["confidence"] for h in other_recent) / len(other_recent))
            other_fits.append(sum(h["fitness"] for h in other_recent) / len(other_recent))

        if not other_divs:
            # Sin comparación posible, usar percentiles propios
            all_divs = sorted(h["feromon_divergence"] for h in hist)
            n = len(all_divs)
            p75 = all_divs[int(n * 0.75)]
            p25 = all_divs[int(n * 0.25)]
            if avg_div >= p75:
                return "explorador"
            elif avg_div <= p25:
                return "explotador"
            return f"Agente-{agent_id_str}"

        mean_other_div  = sum(other_divs)  / len(other_divs)
        mean_other_conf = sum(other_confs) / len(other_confs)
        mean_other_fit  = sum(other_fits)  / len(other_fits)

        # Labels basados en posición relativa a los demás agentes
        # Margen del 10% para evitar empates
        MARGIN = 0.10

        if avg_div > mean_other_div * (1 + MARGIN):
            # Más divergente que el promedio → explorador
            return "explorador"
        elif avg_div < mean_other_div * (1 - MARGIN):
            if avg_conf > mean_other_conf * (1 + MARGIN):
                # Menos divergente Y más confiado → explotador
                return "explotador"
            else:
                # Menos divergente, confianza normal → refinador
                return "refinador"
        elif avg_fit > mean_other_fit * (1 + MARGIN):
            # Fitness superior, divergencia media → dominante
            return "dominante"
        elif avg_fit < mean_other_fit * (1 - MARGIN):
            # Fitness inferior → aprendiendo
            return "aprendiendo"
        else:
            # Sin diferencia significativa aún
            return f"Agente-{agent_id_str}"

    def report(self, step: int) -> str:
        """Genera reporte legible de especialización."""
        lines = [f"\n📊 Especialización del Enjambre (step {step}):"]
        diversities = []
        for k, af in self.current.items():
            label = self._infer_label(k)
            lrf = 0.7 + 0.6 * af.fitness
            lines.append(
                f"  🐜 Agente {k} [{label:22s}]: "
                f"fitness={af.fitness:.3f} | "
                f"div={af.feromon_divergence:.3f} | "
                f"conf={af.confidence:.3f} | "
                f"lr_factor={lrf:.2f}x"
            )
            diversities.append(af.feromon_divergence)
        avg_div = sum(diversities) / len(diversities) if diversities else 0
        lines.append(f"  Diversidad total del enjambre: {avg_div:.3f}")
        return "\n".join(lines)


AGENT_NAMES = {
    0: "léxico",
    1: "semántico",
    2: "generación",
    3: "razonamiento",
    4: "contextual",
    5: "memoria",
    6: "síntesis",
}


@dataclass
class SwarmConfig:
    n_agents: int = 3
    feromon_dim: int = 256
    echolocation_dim: int = 128
    swarm_rounds: int = 2
    infrasound_weight: float = 0.3   # cuánto peso tienen los infrasónidos vs feromonas
    agent_configs: List[AgentConfig] = field(default_factory=list)
    matriarca_config: MatriarcaConfig = field(default_factory=MatriarcaConfig)
    dolphin_config: Optional[DolphinConfig] = None   # None = usar config derivada de agent_configs

    def __post_init__(self):
        if not self.agent_configs:
            for i in range(self.n_agents):
                self.agent_configs.append(AgentConfig(
                    agent_id=i,
                    n_agents=self.n_agents,
                    feromon_dim=self.feromon_dim,
                ))
        if self.dolphin_config is None:
            # Derivar DolphinConfig de los AgentConfigs
            ac = self.agent_configs[0]
            self.dolphin_config = DolphinConfig(
                vocab_size=ac.vocab_size,
                n_embd=ac.n_embd,
                feromon_dim=self.feromon_dim,
                echo_dim=self.echolocation_dim,
                agent_id=self.n_agents,   # id = n_agents (el Delfín es el agente N)
                n_agents=self.n_agents + 1,
            )


# ─── Ecolocalización 🐬 ───────────────────────────────────────────────────────

class EcholocationHead(nn.Module):
    """3 pings simultáneos → embedding 3D del problema."""

    def __init__(self, vocab_size, n_embd, echolocation_dim, feromon_dim):
        super().__init__()
        self.ping_emb = nn.Embedding(vocab_size, n_embd // 4)
        self.ping_topic  = nn.Linear(n_embd // 4, echolocation_dim)
        self.ping_intent = nn.Linear(n_embd // 4, echolocation_dim)
        self.ping_need   = nn.Linear(n_embd // 4, echolocation_dim)
        self.echo_to_feromon = nn.Linear(echolocation_dim * 3, feromon_dim)
        self.norm = nn.LayerNorm(feromon_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.ping_emb(idx).mean(dim=1)
        echo = torch.cat([
            torch.tanh(self.ping_topic(x)),
            torch.tanh(self.ping_intent(x)),
            torch.tanh(self.ping_need(x)),
        ], dim=-1)
        return self.norm(self.echo_to_feromon(echo))


# ─── Pool de Feromonas ────────────────────────────────────────────────────────

class FeromonPool(nn.Module):
    """Agrega señales de todos los agentes en una feromona global."""

    def __init__(self, n_agents, feromon_dim):
        super().__init__()
        self.agent_trust = nn.Parameter(torch.ones(n_agents) / n_agents)
        self.mix = nn.Linear(feromon_dim, feromon_dim, bias=False)

    def forward(self, feromons: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.agent_trust, dim=0)
        pooled = sum(w * f for w, f in zip(weights, feromons))
        return torch.tanh(self.mix(pooled))


# ─── Mezclador de Infrasónidos 🐘 + Feromonas 🐜 ─────────────────────────────

class InfrasoundMixer(nn.Module):
    """
    Mezcla los infrasónidos de la Matriarca con las feromonas del enjambre.
    
    Las feromonas = señal inmediata (qué está pasando ahora)
    Los infrasónidos = sabiduría acumulada (qué aprendimos antes)
    
    El gate aprende cuándo escuchar más a la Matriarca vs al enjambre.
    """
    def __init__(self, feromon_dim: int, infrasound_dim: int):
        super().__init__()
        # Proyectar infrasónidos al mismo espacio que feromonas
        self.infrasound_proj = nn.Linear(infrasound_dim, feromon_dim, bias=False)
        # Gate: ¿cuánto peso dar a los infrasónidos?
        self.gate = nn.Sequential(
            nn.Linear(feromon_dim * 2, feromon_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(feromon_dim)

    def forward(
        self,
        feromon: torch.Tensor,       # [B, feromon_dim]
        infrasound: torch.Tensor,    # [B, infrasound_dim]
    ) -> torch.Tensor:
        infra_proj = self.infrasound_proj(infrasound)   # [B, feromon_dim]
        gate = self.gate(torch.cat([feromon, infra_proj], dim=-1))
        mixed = self.norm(feromon + gate * infra_proj)
        return mixed


# ─── Enjambre Completo 🐜🐘🐬 ─────────────────────────────────────────────────

class LixySwarm(nn.Module):
    """
    Enjambre Lixy-0.1 v2 — Matriarca conectada.

    Flujo completo:
      input
        ↓
      🐬 Ecolocalización → feromon_0
        ↓
      🐘 Matriarca → infrasónidos
        ↓
      InfrasoundMixer(feromon_0, infrasónidos) → feromon_guiada
        ↓
      [Rondas del enjambre]
        agente_0(input, feromon_guiada) → logits_0, feromon_out_0
        agente_1(input, feromon_guiada) → logits_1, feromon_out_1
        agente_2(input, feromon_guiada) → logits_2, feromon_out_2
        ↓
      FeromonPool(feromon_out_0..N) → feromon_nueva
      InfrasoundMixer(feromon_nueva, infrasónidos) → feromon_siguiente_ronda
        ↓
      Agregación por confianza → logits_final
        ↓
      🐘 Matriarca.store(estado_final) — aprende de esta interacción
    """

    def __init__(self, config: SwarmConfig, load_matriarca: bool = True, agent_checkpoint: str = None):
        super().__init__()
        self.config = config

        vocab_size = config.agent_configs[0].vocab_size
        n_embd = config.agent_configs[0].n_embd
        infrasound_dim = config.matriarca_config.infrasound_dim

        # 🐬 DolphinPool — escala con la red (delfines dinámicos)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dolphin = DolphinPool(config.dolphin_config, device=device_str)
        print(f"  🐬 DolphinPool: {self.dolphin.n_dolphins} delfín(es) activos | {sum(p.numel() for p in self.dolphin.parameters())/1e6:.1f}M params")

        # 🐜 Agentes
        self.agents = nn.ModuleList([
            AgentBase(cfg) for cfg in config.agent_configs
        ])

        # Pool de feromonas
        self.feromon_pool = FeromonPool(config.n_agents, config.feromon_dim)

        # 🐘 Mezclador infrasónidos (el cable que faltaba)
        self.infrasound_mixer = InfrasoundMixer(config.feromon_dim, infrasound_dim)

        # Cabezas de confianza por agente
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ) for _ in range(config.n_agents)
        ])

        # Proyección del estado final para la Matriarca
        self.state_to_matriarca = nn.Linear(vocab_size, config.matriarca_config.embd_dim, bias=False)

        # Proyección de infrasónidos → bias de confianza por agente
        # La Matriarca vota qué agente debería tener más peso en la agregación final
        self.matriarca_conf_proj = nn.Sequential(
            nn.Linear(config.matriarca_config.infrasound_dim, config.n_agents, bias=True),
            nn.Tanh(),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"LixySwarm v2: {config.n_agents} agentes, {total_params/1e6:.1f}M params")

        # Cargar checkpoint de agentes si se especifica
        if agent_checkpoint is not None:
            from pathlib import Path as _Path
            ckpt_path = _Path(agent_checkpoint)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
                # Todos los agentes arrancan desde el mismo checkpoint entrenado
                for i, agent in enumerate(self.agents):
                    missing, unexpected = agent.load_state_dict(state, strict=False)
                    if i == 0:
                        print(f"  ✓ Agentes cargados desde {ckpt_path.name} (missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                print(f"  ⚠ Checkpoint de agentes no encontrado: {agent_checkpoint}")

        # 🐘 Matriarca — fuera del módulo nn (no backprop a través de ella)
        self.matriarca: Optional[Matriarca] = None
        if load_matriarca:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            base_mat = Matriarca(config.matriarca_config, device=device)
            # Envolver con MatriarcaEnriched para habilitar legado genético de sectas
            from src.matriarca.matriarca_legacy import MatriarcaEnriched
            self.matriarca = MatriarcaEnriched(base_mat)
            print(f"  🐘 Matriarca conectada: {self.matriarca.memory_count} memorias | 🧬 legados: {self.matriarca.legacy_count}")

        # 🐜 Tracker de especialización
        self.specialization = SpecializationTracker(config.n_agents)
        self._last_fitnesses: Optional[List[AgentFitness]] = None

        # 🐜 Ciclo de vida dinámico — NUEVA ARQUITECTURA
        # Hormiga = Nodo físico. Secta = Especialidad.
        from src.swarm.node_manager import NodeManager, HardwareProfile
        from src.swarm.sect_manager import SectManager
        self.node_manager = NodeManager(matriarca=self.matriarca)
        self.sect_manager = SectManager(
            node_manager=self.node_manager,
            matriarca=self.matriarca,
        )

        # Registrar el nodo local (este proceso) automáticamente
        local_hw = HardwareProfile.from_local()
        self.node_manager.node_joined("local", hardware=local_hw, is_local=True)

        # Sectas iniciales: explorador + refinador (una por tipo de agente actual)
        self._init_default_sects()

        # Compatibilidad: AntLifecycleManager sigue disponible para training
        from src.swarm.ant_lifecycle import AntLifecycleManager
        self.ant_lifecycle = AntLifecycleManager(self, self.matriarca)

    def _init_default_sects(self):
        """Crea las sectas iniciales basándose en el enjambre actual."""
        # Explorador y Refinador — sectas base (el Delfín es enrutador, no secta)
        initial_roles = ["explorador", "refinador"]
        for role in initial_roles:
            sect = self.sect_manager.spawn_sect(role, priority=0.7)
            if sect:
                # Registrar agentes actuales
                for ant in self.agents:
                    self.sect_manager.add_agent_to_sect(sect.sect_id, "local")

    def tick_lifecycle(self, step: int, swarm_diversity: float, n_nodes: int = 1) -> list:
        """
        Tick del ecosistema del enjambre:
        1. NodeManager: prune nodos muertos + registrar contribución
        2. SectManager: ciclo de vida de sectas (nacen/mueren por fitness y diversidad)
        3. AntLifecycleManager: compatibilidad training (agentes individuales)
        4. DolphinPool: escala delfines según nodos conectados
        Retorna lista de todos los eventos.
        """
        events = []

        # Nodos muertos por timeout
        dead_node_events = self.node_manager.prune_dead_nodes()
        events.extend(dead_node_events)

        # Ciclo de vida de sectas
        sect_events = self.sect_manager.tick(step, swarm_diversity)
        events.extend(sect_events)

        # Compatibilidad: ciclo de vida de agentes individuales (training)
        ant_events = self.ant_lifecycle.tick(step, swarm_diversity, n_nodes)
        events.extend(ant_events)

        # Escalar delfines en sync con la red
        dolphin_events = self.scale_dolphins(n_nodes)
        events.extend(dolphin_events)

        # Phase B: actualizar modo de sueño del delfín según diversidad
        for bridge in self.dolphin.dolphins:
            sleep_mode = bridge.update_sleep_mode(swarm_diversity)

        # Phase B real: si hubo inactividad suficiente, consolidar sleep_state.
        for r in self.dolphin.maybe_consolidate_sleep(force=False):
            if r.get("consolidated"):
                events.append({
                    "type": "dolphin_sleep_consolidated",
                    "dolphin_idx": r.get("dolphin_idx", 0),
                    "method": r.get("method", "pca_svd"),
                    "n_contexts": r.get("n_contexts", 0),
                    "explained_variance": r.get("explained_variance", 0.0),
                    "new_norm": r.get("new_norm", 0.0),
                })

        return events

    def route_task(
        self,
        idx: torch.Tensor,
        use_sects: bool = True,
    ):
        """
        Enruta una tarea a través del delfín:
        1. Delfín ecolocaliza → acoustic_map
        2. Router decide la secta adecuada
        3. Retorna RouteDecision + feromon

        Args:
            idx: tokens de input (B, T)
            use_sects: si False, solo retorna feromon sin routing

        Returns:
            (feromon, route_decision, dolphin_info)
        """
        # Forward del delfín primario (construye acoustic_map)
        feromon, dolphin_info = self.dolphin.primary(idx)

        if not use_sects:
            return feromon, None, dolphin_info

        # Enrutar a secta
        sects = self.sect_manager.all_sects()
        route_decision = self.dolphin.primary.route(
            sects=sects,
            matriarca=self.matriarca,
        )

        dolphin_info["route"] = {
            "primary_sect": route_decision.primary_sect,
            "secondary_sects": route_decision.secondary_sects,
            "confidence": route_decision.confidence,
            "mode": route_decision.mode,
            "reason": route_decision.reason,
        }

        return feromon, route_decision, dolphin_info

    def scale_dolphins(self, n_nodes: int) -> list:
        """Escala el pool de delfines según nodos conectados."""
        events = self.dolphin.scale_to_network(n_nodes)
        if events:
            import logging
            log = logging.getLogger("lixy.swarm")
            for e in events:
                if e["type"] == "spawn":
                    log.info(f"🐬 Delfín {e['dolphin_idx']} nacido (red={n_nodes} nodos)")
                elif e["type"] == "retire":
                    log.info(f"🐬 Delfín {e['dolphin_idx']} retirado (red={n_nodes} nodos)")
        return events

    def _get_infrasound(self, feromon: torch.Tensor) -> Optional[torch.Tensor]:
        """Consulta a la Matriarca y obtiene infrasónidos."""
        if self.matriarca is None:
            return None
        state = feromon.mean(dim=0) if feromon.dim() > 1 else feromon
        embd_dim = self.config.matriarca_config.embd_dim
        if state.shape[-1] != embd_dim:
            state = F.interpolate(
                state.unsqueeze(0).unsqueeze(0),
                size=embd_dim, mode='linear', align_corners=False
            ).squeeze()
        infrasound = self.matriarca.emit_infrasound(
            state,
            use_retrieval=True,
            top_k=32,
            update_importance=True,
            importance_delta=0.05,
        )
        return infrasound.to(feromon.device).unsqueeze(0).expand(feromon.shape[0], -1)

    @torch.no_grad()
    def _compute_fitness(
        self,
        logits_list: list,
        feromons_list: list,
        weights: torch.Tensor,
        total_loss: float,
    ) -> List["AgentFitness"]:
        """
        Calcula fitness de cada agente sin gradientes.
        Labels emergen del comportamiento, no se programan.

        Métricas normalizadas contra el enjambre en este step:
        - divergence: relativo al máx del batch (no absoluto)
        - confidence: relativo entre agentes (softmax natural)
        - loss_contribution: relativo al promedio del enjambre
        """
        n = len(self.agents)
        # Pesos de confianza: weights [B, n_agents, 1] → [n_agents]
        conf_per_agent = weights.detach().mean(dim=0).squeeze(-1)  # [n_agents] ya softmaxeado

        # Feromonas apiladas: [n_agents, B, feromon_dim]
        feromons_t = torch.stack([f.detach() for f in feromons_list], dim=0)
        feromon_mean = feromons_t.mean(dim=0)  # [B, feromon_dim]

        avg_loss_per_agent = float(total_loss) / max(n, 1)
        fitnesses = []
        raw_divergences = []
        raw_confidences = []

        for i in range(n):
            fi = F.normalize(feromons_t[i].float(), dim=-1)
            fm = F.normalize(feromon_mean.float(), dim=-1)
            raw_divergences.append(max(0.0, 1.0 - (fi * fm).sum(dim=-1).mean().item()))
            raw_confidences.append(conf_per_agent[i].item())

        # Normalización absoluta — evita que el agente dominante siempre obtenga 1.0
        # Usamos clamp contra rangos absolutos esperados en lugar de normalizar vs el batch
        # divergencia coseno esperada: [0, 1] donde 0=idéntico, 1=opuesto
        # confianza softmax con n agentes: uniforme = 1/n, máximo ~1.0
        uniform_conf = 1.0 / max(n, 1)

        for i in range(n):
            # Divergencia absoluta normalizada al rango [0, 0.5] típico (clamp a [0,1])
            divergence_norm = min(1.0, raw_divergences[i] / 0.5)

            # Confianza relativa al valor uniforme (1/n = sin preferencia)
            # >1.0 significa que es preferido; clamp a [0, 1] para normalizar
            confidence_rel = min(1.0, raw_confidences[i] / (uniform_conf + 1e-8) / n)

            # Loss contribution: qué tan por encima del promedio está este agente
            conf_mean = sum(raw_confidences) / n
            # Centrado en 0.5; valores > promedio → > 0.5
            deviation = (raw_confidences[i] - conf_mean) / (conf_mean + 1e-8)
            loss_contribution = max(0.0, min(1.0, 0.5 + 0.5 * deviation))

            fitness = 0.4 * loss_contribution + 0.35 * divergence_norm + 0.25 * confidence_rel

            fitnesses.append(AgentFitness(
                agent_id=i,
                loss_contribution=loss_contribution,
                feromon_divergence=divergence_norm,
                confidence=confidence_rel,
                fitness=fitness,
            ))

        return fitnesses

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        context_text: str = "",
        store_memory: bool = False,
    ):
        B, T = idx.shape

        # ─── 1. Delfín 🐬 — Ecolocalización + Sueño unihemisférico ───
        feromon, dolphin_info = self.dolphin(idx)  # [B, feromon_dim]

        # ─── 2. Infrasónidos 🐘 y mezcla ───
        infrasound = self._get_infrasound(feromon)
        if infrasound is not None:
            feromon = self.infrasound_mixer(feromon, infrasound)

            # Inyectar estado de sueño del Delfín en la Matriarca
            sleep_signal = dolphin_info.get("sleep_for_matriarca")
            if sleep_signal is not None and store_memory:
                with torch.no_grad():
                    self.matriarca.store_interaction(
                        sleep_signal.detach(),
                        text=f"[dolphin_sleep] {context_text[:100]}",
                        importance=dolphin_info.get("confidence", 0.5),
                    )

        # ─── 3. Rondas del enjambre 🐜 ───
        all_logits = []
        all_hidden = []
        all_round_feromons = []   # para fitness: feromonas de última ronda
        total_loss = 0.0

        # Normalizar feromona inicial para evitar saturación
        feromon = F.normalize(feromon, dim=-1)
        remote_provider = getattr(self, "remote_feromon_provider", None)
        if remote_provider is not None:
            try:
                remote_feromon = remote_provider(feromon)
                if remote_feromon is not None:
                    feromon = F.normalize(
                        remote_feromon.to(device=feromon.device, dtype=feromon.dtype),
                        dim=-1,
                    )
            except Exception as e:
                if not getattr(self, "_remote_feromon_error_logged", False):
                    print(f"  ⚠ Feromona remota ignorada: {e}")
                    self._remote_feromon_error_logged = True

        for round_idx in range(self.config.swarm_rounds):
            round_feromons = []
            round_logits = []

            for agent in self.agents:
                logits, loss, feromon_out = agent(idx, targets=targets, feromon_in=feromon)
                round_feromons.append(feromon_out)
                round_logits.append(logits)
                if loss is not None:
                    total_loss += loss

            # Pool de feromonas + normalizar para estabilidad
            feromon = self.feromon_pool(round_feromons)
            feromon = F.normalize(feromon, dim=-1)

            # Volver a mezclar con infrasónidos en cada ronda
            if infrasound is not None:
                feromon = self.infrasound_mixer(feromon, infrasound)
                feromon = F.normalize(feromon, dim=-1)

            if round_idx == self.config.swarm_rounds - 1:
                all_logits = round_logits
                all_round_feromons = round_feromons  # guardar para fitness

        # ─── 4. Agregación por confianza (con bias de la Matriarca) ───
        # Calcular confianza de cada agente sobre el output
        weights = []
        for i, (logits, conf_head) in enumerate(zip(all_logits, self.confidence_heads)):
            rep = logits.mean(dim=1)  # [B, vocab]
            conf = conf_head(rep[:, :self.config.agent_configs[0].n_embd] if rep.shape[-1] >= self.config.agent_configs[0].n_embd else F.pad(rep, (0, self.config.agent_configs[0].n_embd - rep.shape[-1])))
            weights.append(conf)  # [B, 1]

        weights = torch.stack(weights, dim=1)  # [B, n_agents, 1]

        # Bias de la Matriarca: los infrasónidos orientan qué agente debería tener más peso
        # La Matriarca proyecta su "voto" sobre los n_agents agentes
        if infrasound is not None and self.matriarca is not None:
            with torch.no_grad():
                # infrasound: [B, infrasound_dim] → [B, n_agents] → bias de confianza
                mat_bias = self.matriarca_conf_proj(infrasound)  # [B, n_agents]
                mat_bias = mat_bias.unsqueeze(-1)                # [B, n_agents, 1]
                # Mezcla suave: 80% confidence head propio, 20% voto de la Matriarca
                weights = 0.8 * weights + 0.2 * mat_bias

        weights = F.softmax(weights, dim=1)

        stacked = torch.stack(all_logits, dim=1)  # [B, n_agents, T, vocab]
        aggregated = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, T, vocab]

        # ─── 5. Feedback a la Matriarca 🐘 ───
        if store_memory and self.matriarca is not None and context_text:
            with torch.no_grad():
                state_repr = aggregated.mean(dim=1).mean(dim=0)  # [vocab]
                state_embd = self.state_to_matriarca(state_repr)  # [embd_dim]
                avg_loss = float(total_loss) / max(self.config.n_agents * self.config.swarm_rounds, 1)
                importance = max(0.0, min(1.0, 1.0 - avg_loss / 10.0))
                self.matriarca.store_interaction(
                    state_embd,
                    text=context_text[:200],
                    importance=importance,
                )

        # ─── 6. Fitness de agentes 🐜 (sin gradientes) ───
        if targets is not None:
            fitnesses = self._compute_fitness(all_logits, all_round_feromons, weights, total_loss)
            self._last_fitnesses = fitnesses
        else:
            fitnesses = getattr(self, '_last_fitnesses', None)

        if targets is not None:
            total_loss /= self.config.n_agents * self.config.swarm_rounds

        self._last_feromon = feromon.detach()

        return aggregated, total_loss if targets is not None else None, feromon


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🐜🐘🐬 Test LixySwarm v3 — Delfín como agente real\n")

    cfg = SwarmConfig(n_agents=3, swarm_rounds=2)
    swarm = LixySwarm(cfg, load_matriarca=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    swarm = swarm.to(device)

    x = torch.randint(0, 50304, (2, 64), device=device)

    # Forward con feedback a Matriarca y sueño del Delfín
    logits, loss, feromon = swarm(
        x, targets=x,
        context_text="Test integración Delfín-Matriarca-Enjambre",
        store_memory=True,
    )
    print(f"logits: {logits.shape}")
    print(f"loss: {loss.item():.4f}")
    print(f"feromon: {feromon.shape}")
    if swarm.matriarca:
        print(f"🐘 Memorias en Matriarca: {swarm.matriarca.memory_count}")
    sleep_norm = swarm.dolphin.primary.dolphin.sleep_state.get_state().norm().item()
    print(f"🐬 Sueño del Delfín (norm): {sleep_norm:.4f}")
    print("\n✅ Delfín integrado al enjambre correctamente")
