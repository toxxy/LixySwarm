"""
Lixy-0.1 — RuntimeSession
==========================
Sesión de runtime que mantiene estado entre turnos de conversación.

El problema anterior:
  - Cada llamada a generate_swarm() era stateless
  - La Matriarca recibía feedback solo en el warm-up (1 token)
  - El feromon se refrescaba cada 32 tokens pero SIN Matriarca
  - Sin memoria cross-turn durante inferencia

Esta clase implementa:
  1. Estado de feromon persistente entre turnos (no re-calcular desde cero)
  2. Feedback a la Matriarca al FINAL de cada turno (output completo)
  3. Historial de conversación (últimos N turnos como contexto)
  4. Feromon refresh con Matriarca activa (no solo agentes base)
  5. Penalización de memorias inútiles después de cada sesión

Uso:
    session = RuntimeSession(swarm, enc, device="cuda")
    response = session.turn("Hola, ¿qué es un transformer?")
    response2 = session.turn("Y ¿qué es la atención?")
    session.end_session()   # guarda feedback final a Matriarca
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from src.utils.sampling import sample_token
from src.swarm.dynamic_roles import DynamicRoleAdapter, classify_query


@dataclass
class TurnRecord:
    """Registro de un turno de conversación."""
    user_input: str
    response: str
    timestamp: float
    infrasound_norm: float
    feromon_norm: float
    n_tokens: int
    task_type: str = "conversacional"   # tipo de tarea detectado


@dataclass
class SessionStats:
    """Estadísticas acumuladas de la sesión."""
    total_turns: int = 0
    total_tokens: int = 0
    avg_infrasound_norm: float = 0.0
    avg_feromon_norm: float = 0.0
    matriarca_memories_start: int = 0
    matriarca_memories_end: int = 0
    session_duration_s: float = 0.0


class RuntimeSession:
    """
    Sesión de runtime con estado persistente entre turnos.
    
    La Matriarca acumula conocimiento de la conversación completa
    y orienta al enjambre con contexto cross-turn.
    """

    # Cuántos tokens generar antes de re-consultar la Matriarca
    FEROMON_REFRESH_INTERVAL = 32
    # Máx turnos a mantener en el contexto acumulado
    MAX_HISTORY_TURNS = 5

    def __init__(
        self,
        swarm,                    # LixySwarm
        enc,                      # tiktoken encoder
        device: str = "cuda",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.3,
        session_id: Optional[str] = None,
        session_file: Optional[str] = None,   # ruta para persistencia a disco
        verbose: bool = True,
    ):
        self.swarm = swarm
        self.enc = enc
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.session_id = session_id or f"session_{int(time.time())}"
        self.session_file = session_file
        self.verbose = verbose

        # Estado del enjambre entre turnos
        self._cached_feromon: Optional[torch.Tensor] = None
        self._feromon_fresh_at_step: int = 0
        # Feromona remota inyectada desde peers de la red P2P
        self._remote_feromon_injection: Optional[torch.Tensor] = None

        # Historial de conversación
        self.history: list[TurnRecord] = []

        # Stats
        self.stats = SessionStats()
        self.stats.matriarca_memories_start = (
            swarm.matriarca.memory_count if swarm.matriarca else 0
        )
        self._session_start = time.time()
        # Registro de índices de memorias accedidas durante la sesión
        self._accessed_memory_indices: set[int] = set()
        # Últimos índices de memoria guardados (para retroactive feedback)
        self._last_memory_idx: set[int] = set()

        # Adaptador de roles dinámicos
        self._role_adapter = DynamicRoleAdapter(
            n_agents=swarm.config.n_agents,
            verbose=verbose,
        )

        # Cargar historial persistido si existe
        self._load_history()

        if verbose:
            n_loaded = len(self.history)
            print(f"🧠 RuntimeSession iniciada: {self.session_id}")
            if n_loaded > 0:
                print(f"  📥 Historial cargado: {n_loaded} turnos previos")
            if swarm.matriarca:
                print(f"  🐘 Matriarca: {swarm.matriarca.memory_count} memorias activas")

    # ─── Persistencia a disco ─────────────────────────────────────────────────

    def _load_history(self):
        """
        Carga el historial de conversación desde disco si existe.
        Permite continuar una sesión entre reinicios.
        """
        import json
        from pathlib import Path

        if not self.session_file:
            return

        path = Path(self.session_file)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            # Restaurar historial de turnos
            loaded = []
            for t in data.get("history", []):
                loaded.append(TurnRecord(
                    user_input=t["user_input"],
                    response=t["response"],
                    timestamp=t["timestamp"],
                    infrasound_norm=t.get("infrasound_norm", 0.0),
                    feromon_norm=t.get("feromon_norm", 0.0),
                    n_tokens=t.get("n_tokens", 0),
                    task_type=t.get("task_type", "conversacional"),
                ))
            self.history = loaded

            # Restaurar session_id si existe
            saved_id = data.get("session_id")
            if saved_id:
                self.session_id = saved_id

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ No se pudo cargar historial: {e}")

    def _save_history(self):
        """
        Persiste el historial de conversación a disco.
        Se llama después de cada turno y en end_session().
        """
        import json
        from pathlib import Path

        if not self.session_file:
            return

        path = Path(self.session_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_turns": self.stats.total_turns,
            "task_distribution": self._role_adapter.task_distribution(),
            "history": [
                {
                    "user_input": t.user_input,
                    "response": t.response,
                    "timestamp": t.timestamp,
                    "infrasound_norm": t.infrasound_norm,
                    "feromon_norm": t.feromon_norm,
                    "n_tokens": t.n_tokens,
                    "task_type": t.task_type,
                }
                for t in self.history[-100:]   # últimos 100 turnos máx
            ],
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ No se pudo guardar historial: {e}")

    # ─── Contexto Acumulado ───────────────────────────────────────────────────

    def _build_context(self, current_input: str) -> str:
        """
        Construye el prompt con contexto de la conversación.
        Los últimos N turnos se incluyen como contexto para la Matriarca.
        """
        recent = self.history[-self.MAX_HISTORY_TURNS:]
        if not recent:
            return current_input

        lines = []
        for turn in recent:
            lines.append(f"Human: {turn.user_input[:100]}")
            lines.append(f"Lixy: {turn.response[:150]}")
        lines.append(f"Human: {current_input}")
        return "\n".join(lines)

    # ─── Turno Principal ──────────────────────────────────────────────────────

    # ─── Feedback explícito del usuario ─────────────────────────────────────

    # Señales positivas (el usuario indica que la respuesta anterior fue útil)
    _POSITIVE_SIGNALS = {
        'es': [
            'gracias', 'perfecto', 'exacto', 'correcto', 'genial', 'excelente',
            'muy bien', 'bien', 'sí', 'si', 'claro', 'entendí', 'entendido',
            'tiene sentido', 'funciona', 'funcionó', 'lo logré', 'listo',
        ],
        'en': [
            'thanks', 'thank you', 'perfect', 'exactly', 'correct', 'great',
            'excellent', 'very good', 'good', 'yes', 'got it', 'understood',
            'makes sense', 'works', 'worked', 'done', 'nice',
        ],
    }

    # Señales negativas (el usuario indica que la respuesta anterior fue inútil)
    _NEGATIVE_SIGNALS = {
        'es': [
            'no', 'mal', 'incorrecto', 'está mal', 'eso no',
            'no sirve', 'no funciona',
            'confuso', 'equivocado', 'error', 'eso no era',
        ],
        'en': [
            "no", "wrong", "incorrect", "that's wrong", "not right",
            "doesn't work", "doesn't make sense", "confused", "error",
            "that's not", "not what", "try again", "not correct",
        ],
    }

    def _detect_explicit_feedback(self, user_input: str) -> float:
        """
        Detecta si el mensaje del usuario contiene feedback explícito sobre
        la respuesta anterior.

        Examina los primeros ~60 chars del mensaje para detectar señales
        positivas (+1.0) o negativas (-1.0). Si no hay señal clara → 0.0.

        La detección es intencional-mente conservadora: solo actúa cuando
        el mensaje COMIENZA con una señal clara (evita falsos positivos).

        Returns:
            +1.0 = señal positiva clara
            -1.0 = señal negativa clara
             0.0 = neutral / sin señal
        """
        if not self.history:   # no hay turno anterior que valorar
            return 0.0

        text = user_input.strip().lower()[:80]

        # Positivo — verificar que COMIENZA con la señal
        for signals in self._POSITIVE_SIGNALS.values():
            for signal in signals:
                # Acepta "signal" al inicio, con o sin puntuación previa
                clean = text.lstrip('¡!¿ ')
                if clean.startswith(signal):
                    # Evitar "bien" dentro de "bienestar" etc. — verificar límite de palabra
                    end_pos = len(signal)
                    if end_pos >= len(clean) or not clean[end_pos].isalpha():
                        return +1.0

        # Negativo — igual, solo al inicio
        for signals in self._NEGATIVE_SIGNALS.values():
            for signal in signals:
                if text.startswith(signal):
                    # Excepción: frases que empiezan con "no" pero son neutrales
                    if signal == 'no':
                        neutral_continuations = [
                            'sé ', 'se ', 'creo', 'tengo', 'recuerdo', 'puedo',
                            'entiendo', 'entiendo.', 'entiendo,', 'entendí', 'entendí ', 'estoy', 'hay',
                            'entend', 'entiend',
                            'lo ', 'me ', 'te ', 'le ', 'es ', 'era ',
                            'tiene', 'quiero', 'necesito', 'importa',
                        ]
                        rest = text[3:] if len(text) > 3 else ''  # text after 'no '
                        if any(rest.startswith(w.strip()) for w in neutral_continuations):
                            continue
                        # "no, ..." con coma sí es feedback negativo
                        # "no" solo sí es feedback negativo
                        # pero "no sé", "no entiendo" son neutrales
                    return -1.0

        return 0.0

    def _apply_explicit_feedback(
        self,
        feedback_score: float,
        positive_delta: float = 0.15,
        negative_delta: float = -0.20,
    ):
        """
        Aplica el feedback explícito del usuario a la última memoria almacenada.

        +feedback: sube importancia de la última memoria +15%
        -feedback: baja importancia de la última memoria -20%

        Solo actúa si hay memorias del turno anterior disponibles.
        """
        if not self._last_memory_idx or self.swarm.matriarca is None:
            return

        if feedback_score == 0.0:
            return

        delta = positive_delta if feedback_score > 0 else negative_delta
        indices = torch.tensor(list(self._last_memory_idx), dtype=torch.long)
        self.swarm.matriarca.bank.update_importance(indices, delta=delta)

        if self.verbose:
            sign = '✅ +' if delta > 0 else '⛔ '
            print(
                f"  {sign}Feedback explícito: {delta:+.0%} en "
                f"{len(self._last_memory_idx)} memorias del turno anterior"
            )

    def inject_remote_feromon(
        self,
        remote_feromon: torch.Tensor,
        blend_weight: float = 0.25,
    ):
        """
        Inyecta una feromona remota (de un peer de la red P2P) para ser
        mezclada con la feromona local en el próximo warm-up.

        El blend es conservador (25% por defecto) para que los peers
        orienten sutilmente sin dominar la generación local.

        Args:
            remote_feromon: tensor [feromon_dim] del peer remoto
            blend_weight:   peso del peer (0.0–1.0), default 0.25
        """
        if remote_feromon is None:
            return
        rf = remote_feromon.detach().cpu()
        if rf.shape != torch.Size([self.swarm.config.feromon_dim]):
            # Redimensionar si hay mismatch de dimensión
            try:
                rf = F.interpolate(
                    rf.float().unsqueeze(0).unsqueeze(0),
                    size=self.swarm.config.feromon_dim,
                    mode='linear', align_corners=False,
                ).squeeze()
            except Exception:
                return  # silencioso si no se puede interpolar
        self._remote_feromon_injection = (rf, blend_weight)
        if self.verbose:
            print(f"  🌐 Feromona remota inyectada (blend={blend_weight:.0%})")

    @torch.no_grad()
    def turn(
        self,
        user_input: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        store_memory: bool = True,
        record_history: bool = True,
        update_runtime_state: bool = True,
        use_memory: bool = True,
    ) -> str:
        """
        Procesa un turno de conversación.
        
        Mejoras vs generate_swarm():
        - Feromon se reutiliza del turno anterior (warm-start)
        - Refresh del feromon incluye Matriarca activa
        - Feedback a Matriarca al FINAL con output completo
        - Contexto cross-turn en la consulta a la Matriarca
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        k = top_k or self.top_k

        swarm = self.swarm
        enc = self.enc
        device = self.device
        block_size = swarm.config.agent_configs[0].block_size
        n_embd = swarm.config.agent_configs[0].n_embd

        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" else nullcontext()
        )

        # Feedback explícito: detectar si el usuario valora la última respuesta
        # Se hace ANTES de clasificar la query para no contaminar el rol dinámico
        if store_memory:
            explicit_feedback = self._detect_explicit_feedback(user_input)
            if explicit_feedback != 0.0:
                self._apply_explicit_feedback(explicit_feedback)

        # Clasificar query y adaptar temperatura + roles de agentes
        task_profile, role_weights = self._role_adapter.get_weights_for_query(
            user_input,
            base_weights=None,  # sin pesos base aún — se aplican post-warm-up
        )
        # Temperatura: override dinámico si el caller no especificó temperatura
        temp = temperature if temperature is not None else task_profile.temperature

        # Contexto acumulado para la Matriarca
        context_with_history = self._build_context(user_input)

        tokens = enc.encode(user_input)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        infrasound_norms: list[float] = []
        feromon_norms: list[float] = []

        with torch.no_grad(), ctx:

            # ── Paso 0: Warm-up con Matriarca + contexto cross-turn ──────────
            # Si tenemos feromon previo, lo usamos como punto de partida
            # pero igualmente consultamos la Matriarca con el nuevo contexto
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

            logits, _, feromon = swarm(
                x_cond,
                context_text=context_with_history[:200],
                store_memory=False,    # no guardar aún — esperamos al final del turno
                update_runtime_state=update_runtime_state,
                update_memory_importance=store_memory,
                use_memory=use_memory,
            )

            # Si hay feromon previo, mezclar (warm-start: 70% nuevo, 30% memoria)
            if self._cached_feromon is not None:
                prev = self._cached_feromon
                if prev.shape == feromon.shape:
                    feromon = 0.7 * feromon + 0.3 * prev
                    feromon = F.normalize(feromon, dim=-1)

            # Aplicar feromona remota inyectada desde peers P2P (si existe)
            if self._remote_feromon_injection is not None:
                remote_f, blend_w = self._remote_feromon_injection
                remote_f = remote_f.to(device)
                if remote_f.shape == feromon.shape:
                    feromon = (1 - blend_w) * feromon + blend_w * remote_f
                    feromon = F.normalize(feromon, dim=-1)
                self._remote_feromon_injection = None  # consumir

            cached_feromon = feromon.detach()
            self._cached_feromon = cached_feromon

            # Stats warm-up
            if use_memory and swarm.matriarca is not None:
                infra = swarm._get_infrasound(
                    cached_feromon, update_importance=store_memory
                )
                if infra is not None:
                    infrasound_norms.append(infra.norm().item())
                # Registrar qué memorias se accedieron en este warm-up
                if store_memory:
                    self._track_accessed_memories()
            feromon_norms.append(cached_feromon.norm().item())

            # ── Generación token a token ──────────────────────────────────────
            for step_i in range(1, max_tokens):
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

                # Refresh periódico con Matriarca activa
                should_refresh = (step_i % self.FEROMON_REFRESH_INTERVAL == 0)
                if should_refresh:
                    _, _, new_feromon = swarm(
                        x_cond,
                        context_text=context_with_history[:100],
                        store_memory=False,
                        update_runtime_state=update_runtime_state,
                        update_memory_importance=store_memory,
                        use_memory=use_memory,
                    )
                    # Blend: 60% nuevo contexto, 40% feromon acumulado
                    cached_feromon = 0.6 * new_feromon.detach() + 0.4 * cached_feromon
                    cached_feromon = F.normalize(cached_feromon, dim=-1)
                    self._cached_feromon = cached_feromon

                    if use_memory and swarm.matriarca is not None:
                        infra = swarm._get_infrasound(
                            cached_feromon, update_importance=store_memory
                        )
                        if infra is not None:
                            infrasound_norms.append(infra.norm().item())
                        if store_memory:
                            self._track_accessed_memories()
                    feromon_norms.append(cached_feromon.norm().item())

                # Forward de agentes base con feromon fijo
                all_logits = []
                all_conf = []
                for agent, conf_head in zip(swarm.agents, swarm.confidence_heads):
                    ag_logits, _, _ = agent(x_cond, feromon_in=cached_feromon)
                    rep = ag_logits.mean(dim=1)  # [B, vocab]
                    rep_proj = (
                        rep[:, :n_embd] if rep.shape[-1] >= n_embd
                        else F.pad(rep, (0, n_embd - rep.shape[-1]))
                    )
                    conf = conf_head(rep_proj)
                    all_logits.append(ag_logits)
                    all_conf.append(conf)

                # Confianza base de los agentes
                base_conf = F.softmax(torch.cat(all_conf, dim=-1), dim=-1)  # [B, n_agents]

                # Aplicar bias dinámico de roles (35% peso al perfil de tarea)
                rw = role_weights.to(device)  # [n_agents]
                weights = 0.65 * base_conf + 0.35 * rw.unsqueeze(0)  # broadcast [B, n_agents]
                weights = F.softmax(weights, dim=-1)

                logits = sum(
                    w.unsqueeze(-1).unsqueeze(-1) * l
                    for w, l in zip(weights.unbind(dim=-1), all_logits)
                )

                # Sampling — rep_penalty + top_k + top_p
                next_token = sample_token(
                    logits,
                    generated_ids=x,
                    temperature=temp,
                    top_k=k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    recent_penalty=self.repetition_penalty * 2.5,
                    recent_window=8,
                )
                x = torch.cat((x, next_token), dim=1)

                if next_token.item() == enc._special_tokens.get("<|endoftext|>", -1):
                    break

        # ── Decodificar respuesta ─────────────────────────────────────────────
        full_output = enc.decode(x[0].tolist())
        response = full_output[len(enc.decode(tokens)):]

        # ── Feedback post-turno a la Matriarca 🐘 ────────────────────────────
        # Ahora que tenemos el output completo, lo almacenamos con contexto rico
        if store_memory and swarm.matriarca is not None:
            self._store_turn_to_matriarca(
                user_input=user_input,
                response=response,
                feromon=self._cached_feromon,
                full_context=context_with_history,
            )

        # ── Registrar turno ───────────────────────────────────────────────────
        avg_infra = sum(infrasound_norms) / len(infrasound_norms) if infrasound_norms else 0.0
        avg_fero = sum(feromon_norms) / len(feromon_norms) if feromon_norms else 0.0
        n_new_tokens = x.size(1) - len(tokens)

        if record_history:
            record = TurnRecord(
                user_input=user_input,
                response=response,
                timestamp=time.time(),
                infrasound_norm=avg_infra,
                feromon_norm=avg_fero,
                n_tokens=n_new_tokens,
                task_type=task_profile.task_type,
            )
            self.history.append(record)
            self.stats.total_turns += 1
            self.stats.total_tokens += n_new_tokens

            # Persistir historial después de cada turn.
            self._save_history()

        if self.verbose:
            print(f"  [🐘 infra={avg_infra:.3f} | 🐜 feromon={avg_fero:.3f} | tokens={n_new_tokens} | 🧠 {task_profile.task_type}]")

        return response

    def _track_accessed_memories(self):
        """
        Registra qué memorias fueron accedidas en el último forward pass.
        Usa last_access timestamp para detectar memorias recién consultadas.
        """
        if self.swarm.matriarca is None:
            return
        bank = self.swarm.matriarca.bank
        for i, meta in enumerate(bank.metadata):
            last = meta.get("last_access")
            if last is not None and last >= self._session_start:
                self._accessed_memory_indices.add(i)

    def _compute_response_importance(
        self,
        user_input: str,
        response: str,
        feromon: torch.Tensor,
    ) -> float:
        """
        Importancia dinámica basada en coherencia real de la respuesta.

        Métricas utilizadas:
        1. Longitud normalizada (más texto = más información potencial)
        2. Tipo-token ratio (TTR): diversidad léxica — alta TTR = baja repetición
        3. Coherencia semántica: solapamiento de tokens input/output
           (si la respuesta habla de lo mismo que la pregunta = relevante)
        4. Penalización por repetición de n-gramas (2-gramas repetidos = texto degenerado)
        5. Continuidad temática: si el turno anterior era sobre el mismo tema
        """
        if not response or not response.strip():
            return 0.1

        tokens = response.split()
        n = len(tokens)
        if n == 0:
            return 0.1

        # ──  1. Factor de longitud ────────────────────────────────────────────
        # Saturación en 150 tokens: el modelo actual genera máximo ~200
        length_factor = min(1.0, n / 150)

        # ── 2. Tipo-token ratio (TTR) ─────────────────────────────────────────
        # Textos no repetitivos tienen TTR alto. Normalizar por longitud
        # para que no penalice textos largos por definición.
        # RTTR = |tipos| / sqrt(|tokens|)  (Root TTR, insensible a longitud)
        unique_tokens = set(t.lower() for t in tokens)
        rttr = len(unique_tokens) / max(1.0, n ** 0.5)
        # RTTR esperado en texto natural ~3-8. Normalizar a [0,1]
        ttr_factor = min(1.0, rttr / 5.0)

        # ── 3. Penalización por repetición de 2-gramas ───────────────────────
        # Un loop de generación produce bigramas muy repetidos
        if n >= 4:
            bigrams = [(tokens[i].lower(), tokens[i+1].lower()) for i in range(n-1)]
            unique_bigrams = set(bigrams)
            bigram_repeat_ratio = 1.0 - (len(unique_bigrams) / max(1, len(bigrams)))
            # bigram_repeat_ratio ~ 0 = no repetición, ~1 = todo repetido
            repetition_penalty = max(0.0, 1.0 - 2.0 * bigram_repeat_ratio)
        else:
            repetition_penalty = 0.8  # texto muy corto: penalizar moderadamente

        # ── 4. Coherencia semántica input→output ────────────────────────────
        # Solapamiento de vocabulario significativo entre pregunta y respuesta
        # (stopwords excluidas para ser más preciso)
        STOPWORDS = {
            'el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'y', 'a',
            'que', 'es', 'se', 'por', 'con', 'no', 'lo', 'su', 'le', 'al',
            'the', 'a', 'an', 'of', 'in', 'and', 'to', 'is', 'that', 'for',
            'it', 'was', 'on', 'are', 'as', 'be', 'this', 'with', 'his',
        }
        input_words = {w.lower() for w in user_input.split() if w.lower() not in STOPWORDS and len(w) > 2}
        output_words = {w.lower() for w in tokens if w.lower() not in STOPWORDS and len(w) > 2}
        if input_words and output_words:
            overlap = len(input_words & output_words)
            # Jaccard: overlap / union. Respuesta ON-TOPIC si Jaccard > 0.1
            jaccard = overlap / max(1, len(input_words | output_words))
            coherence_factor = min(1.0, jaccard * 5.0)  # normalizar: 0.2 jaccard = 1.0
        else:
            coherence_factor = 0.5  # sin información suficiente

        # ── 5. Continuidad temática con turno anterior ───────────────────────
        # Si el usuario retomó el tema del turno anterior, la memoria anterior
        # fue útil. Guardaremos esto para feedback retroactivo.
        topical_continuity = 0.5  # neutral por defecto
        if self.history:
            prev = self.history[-1]
            prev_words = {w.lower() for w in prev.user_input.split() if len(w) > 2}
            curr_words = {w.lower() for w in user_input.split() if len(w) > 2}
            if prev_words and curr_words:
                prev_overlap = len(prev_words & curr_words) / max(1, len(prev_words | curr_words))
                topical_continuity = min(1.0, prev_overlap * 4.0)

        # ── Composición final ────────────────────────────────────────────────
        # Pesos: repetition_penalty más importante (texto degradado = inutil)
        importance = (
            0.30 * length_factor
            + 0.25 * ttr_factor
            + 0.25 * repetition_penalty
            + 0.15 * coherence_factor
            + 0.05 * topical_continuity
        )
        importance = max(0.05, min(1.0, importance))

        # Logging debug (solo si verbose)
        if self.verbose:
            print(
                f"  🧠 importancia: {importance:.3f} "
                f"[len={length_factor:.2f} ttr={ttr_factor:.2f} "
                f"rep={repetition_penalty:.2f} coh={coherence_factor:.2f} "
                f"cont={topical_continuity:.2f}]"
            )

        return importance

    def _retroactive_feedback(
        self,
        user_input: str,
        importance_delta: float = 0.1,
    ):
        """
        Feedback retroactivo: si el usuario sigue en el mismo tema que el turno
        anterior, la última memoria guardada fue útil → subir su importancia.

        Esto cierra el loop de aprendizaje: la Matriarca aprende qué memorias
        son realmente útiles para la conversación (no solo por métricas locales).
        """
        if not self.history or not self._last_memory_idx:
            return

        prev = self.history[-1]
        prev_words = {w.lower() for w in prev.user_input.split() if len(w) > 2}
        curr_words = {w.lower() for w in user_input.split() if len(w) > 2}

        if not prev_words or not curr_words:
            return

        overlap_ratio = len(prev_words & curr_words) / max(1, len(prev_words | curr_words))
        if overlap_ratio > 0.15:  # umbral: 15% de palabras en común = mismo tema
            # Subir importancia de la memoria del turno anterior
            self.swarm.matriarca.bank.update_importance(
                torch.tensor(list(self._last_memory_idx), dtype=torch.long),
                delta=importance_delta * overlap_ratio,
            )
            if self.verbose:
                print(
                    f"  🔄 Feedback retroactivo: +{importance_delta * overlap_ratio:.3f} "
                    f"a {len(self._last_memory_idx)} memorias (solapamiento={overlap_ratio:.2f})"
                )

    def _store_turn_to_matriarca(
        self,
        user_input: str,
        response: str,
        feromon: torch.Tensor,
        full_context: str,
    ):
        """
        Almacena el turno completo en la Matriarca.
        La importancia se calcula dinámicamente por coherencia y calidad.
        También aplica feedback retroactivo al turno anterior si hubo continuidad temática.
        """
        swarm = self.swarm
        device = self.device

        # Feedback retroactivo al turno anterior (antes de agregar el nuevo)
        self._retroactive_feedback(user_input, importance_delta=0.08)

        # Calcular importancia dinámica basada en coherencia real
        importance = self._compute_response_importance(user_input, response, feromon)

        # Texto a almacenar: prompt + primeros 150 chars de respuesta
        mem_text = f"[runtime] Q: {user_input[:100]} A: {response[:150]}"

        # Usar feromon actual como embedding de la interacción
        embd_dim = swarm.config.matriarca_config.embd_dim
        state = feromon.mean(dim=0) if feromon.dim() > 1 else feromon

        if state.shape[-1] != embd_dim:
            state = F.interpolate(
                state.float().unsqueeze(0).unsqueeze(0),
                size=embd_dim, mode='linear', align_corners=False,
            ).squeeze()

        # Registrar el índice de la nueva memoria para posible feedback retroactivo futuro
        mem_idx_before = swarm.matriarca.bank.size

        with torch.no_grad():
            swarm.matriarca.store_interaction(
                state.to(device),
                text=mem_text,
                importance=importance,
            )

        # Guardar índices de las memorias recién agregadas para retroactive feedback futuro
        mem_idx_after = swarm.matriarca.bank.size
        self._last_memory_idx = set(range(mem_idx_before, mem_idx_after))

    # ─── Fin de Sesión ────────────────────────────────────────────────────────

    def end_session(self, save_matriarca: bool = True):
        """
        Cierra la sesión y guarda el estado de la Matriarca.
        
        También almacena un resumen de la conversación completa
        como memoria de alta importancia.
        """
        swarm = self.swarm

        if not self.history:
            return

        # Resumen de la sesión para la Matriarca
        if swarm.matriarca is not None and self._cached_feromon is not None:
            turns_summary = " | ".join(
                f"'{t.user_input[:40]}'" for t in self.history[-3:]
            )
            session_summary = f"[sesión {self.session_id}] {len(self.history)} turnos: {turns_summary}"

            embd_dim = swarm.config.matriarca_config.embd_dim
            state = self._cached_feromon.mean(dim=0)
            if state.shape[-1] != embd_dim:
                state = F.interpolate(
                    state.float().unsqueeze(0).unsqueeze(0),
                    size=embd_dim, mode='linear', align_corners=False,
                ).squeeze()

            with torch.no_grad():
                swarm.matriarca.store_interaction(
                    state.to(self.device),
                    text=session_summary,
                    importance=0.85,   # los resúmenes de sesión son importantes
                )

        # Guardar Matriarca
        if save_matriarca and swarm.matriarca is not None:
            # Penalizar memorias no usadas en esta sesión
            # Las memorias que nunca fueron recuperadas bajan importancia (-2%)
            # Esto mantiene el banco limpio y da más peso a memorias útiles
            if swarm.matriarca.bank.size > 0:
                used = torch.tensor(
                    list(self._accessed_memory_indices), dtype=torch.long
                ) if self._accessed_memory_indices else torch.tensor([], dtype=torch.long)
                n_total = swarm.matriarca.bank.size
                n_used = len(self._accessed_memory_indices)
                swarm.matriarca.penalize_unused(
                    top_k_used=used,
                    all_indices=range(n_total),
                    penalty=-0.02,
                )
                if self.verbose:
                    print(f"  🧠 Penalizadas: {n_total - n_used} memorias no usadas (-2% importancia)")
                    print(f"  🧠 Protegidas:  {n_used} memorias accedidas esta sesión")

            swarm.matriarca.save()

        # Persistir historial final
        self._save_history()

        # Estadísticas finales
        self.stats.matriarca_memories_end = (
            swarm.matriarca.memory_count if swarm.matriarca else 0
        )
        self.stats.session_duration_s = time.time() - self._session_start
        self.stats.avg_infrasound_norm = (
            sum(t.infrasound_norm for t in self.history) / len(self.history)
        )
        self.stats.avg_feromon_norm = (
            sum(t.feromon_norm for t in self.history) / len(self.history)
        )

        if self.verbose:
            new_mems = self.stats.matriarca_memories_end - self.stats.matriarca_memories_start
            print(f"\n🧠 Sesión terminada: {self.session_id}")
            print(f"  Turnos: {self.stats.total_turns} | Tokens: {self.stats.total_tokens}")
            print(f"  Nuevas memorias Matriarca: +{new_mems}")
            print(f"  Duración: {self.stats.session_duration_s:.1f}s")

        return self.stats

    def reset_feromon(self):
        """Reinicia el feromon (para cambio de tema brusco)."""
        self._cached_feromon = None
        self._remote_feromon_injection = None
        self._accessed_memory_indices.clear()
        if self.verbose:
            print("  🔄 Feromon y contexto de memoria reiniciados")


# ─── Integración con generate.py (modo interactivo mejorado) ─────────────────

def interactive_session(swarm, enc, device: str = "cuda"):
    """
    Modo interactivo que usa RuntimeSession para estado persistente.
    Reemplaza interactive_mode() en generate.py.
    """
    session = RuntimeSession(swarm, enc, device=device, verbose=True)
    print()
    print("🐜🐘🐬 Lixy-0.1 — Enjambre Completo — Sesión Persistente")
    print("   (escribe 'salir' para terminar, 'reset' para limpiar feromon)")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego!")
            break

        if user_input.lower() in ("salir", "exit", "quit"):
            print("👋 Hasta luego!")
            break

        if user_input.lower() == "reset":
            session.reset_feromon()
            print("  [🔄 Contexto de feromon reiniciado]\n")
            continue

        if user_input.lower() == "stats":
            print(f"  📊 Turnos: {session.stats.total_turns}")
            print(f"  📊 Tokens: {session.stats.total_tokens}")
            if swarm.matriarca:
                print(f"  🐘 Memorias: {swarm.matriarca.memory_count}")
            print()
            continue

        if not user_input:
            continue

        print("Lixy: ", end="", flush=True)
        response = session.turn(user_input)
        print(response)
        print()

    session.end_session(save_matriarca=True)


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧪 Test RuntimeSession...\n")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.swarm.orchestrator import LixySwarm, SwarmConfig
    from src.utils.tokenizer import get_gpt2_encoding

    enc = get_gpt2_encoding()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SwarmConfig(n_agents=3, swarm_rounds=2)
    swarm = LixySwarm(cfg, load_matriarca=True).to(device)
    mems_before = swarm.matriarca.memory_count if swarm.matriarca else 0

    session = RuntimeSession(swarm, enc, device=device, max_new_tokens=30)

    print("── Turno 1 ──")
    r1 = session.turn("Hola, ¿qué es un transformer?")
    print(f"Respuesta: {r1[:80]}...")

    print("\n── Turno 2 (contexto cross-turn) ──")
    r2 = session.turn("¿Y cómo funciona la atención?")
    print(f"Respuesta: {r2[:80]}...")

    print("\n── Fin de sesión ──")
    stats = session.end_session()

    mems_after = swarm.matriarca.memory_count if swarm.matriarca else 0
    print(f"\n✅ Test completado")
    print(f"  Memorias antes: {mems_before} → después: {mems_after} (+{mems_after - mems_before})")
