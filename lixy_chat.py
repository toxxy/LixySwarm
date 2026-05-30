#!/usr/bin/env python3
"""
lixy_chat.py — Chat CLI para Lixy-0.1 🐜🐘🐬
==============================================
Interface de chat interactiva para Emmanuel.

Uso:
    python3 lixy_chat.py                        # carga swarm_best.pt automáticamente
    python3 lixy_chat.py --checkpoint ckpt.pt   # checkpoint específico
    python3 lixy_chat.py --no-color             # sin colores (terminales básicas)
    python3 lixy_chat.py --debug                # muestra stats extendidos

Comandos en el chat:
    /ayuda          — muestra esta ayuda
    /status         — estado del enjambre y Matriarca
    /reset          — reinicia el contexto de conversación
    /guardar        — guarda la sesión manualmente
    /stats          — distribución de tareas de la sesión
    /salir          — salir y guardar sesión
    Ctrl+C          — salir gracefully

Teclas de agentes (modo debug):
    Muestra qué agente dominó cada respuesta y con qué peso.
"""

import sys
import os
import argparse
import signal
import time
from pathlib import Path

# ── Colores ANSI ──────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    RED     = "\033[91m"
    WHITE   = "\033[97m"

    @classmethod
    def disable(cls):
        for attr in ['RESET','BOLD','DIM','CYAN','GREEN','YELLOW','MAGENTA','BLUE','RED','WHITE']:
            setattr(cls, attr, "")

# ── Spinner para loading ──────────────────────────────────────────────────────
def spinner(msg: str):
    """Muestra un spinner mientras carga."""
    frames = ["⣾","⣽","⣻","⢿","⡿","⣟","⣯","⣷"]
    i = 0
    print(f"\r{C.DIM}{msg} {frames[i]}{C.RESET}", end="", flush=True)
    return frames, i

# ── Bienvenida ────────────────────────────────────────────────────────────────
BANNER = f"""{C.CYAN}{C.BOLD}
  ██╗     ██╗██╗  ██╗██╗   ██╗
  ██║     ██║╚██╗██╔╝╚██╗ ██╔╝
  ██║     ██║ ╚███╔╝  ╚████╔╝
  ██║     ██║ ██╔██╗   ╚██╔╝
  ███████╗██║██╔╝ ██╗   ██║
  ╚══════╝╚═╝╚═╝  ╚═╝   ╚═╝{C.RESET}
{C.MAGENTA}  🐜🐘🐬  LixySwarm v1 — AntElephantDolphin{C.RESET}
"""

HELP_TEXT = f"""
{C.BOLD}Comandos disponibles:{C.RESET}
  {C.CYAN}/ayuda{C.RESET}     — esta ayuda
  {C.CYAN}/status{C.RESET}    — estado del enjambre (memorias, sueño, especialización)
  {C.CYAN}/reset{C.RESET}     — reinicia contexto de conversación (feromon + historial)
  {C.CYAN}/guardar{C.RESET}   — guarda la sesión manualmente
  {C.CYAN}/stats{C.RESET}     — distribución de tipos de tarea en esta sesión
  {C.CYAN}/salir{C.RESET}     — salir y guardar sesión
  {C.CYAN}Ctrl+C{C.RESET}     — salir gracefully

{C.DIM}Tipos de tarea detectados automáticamente:{C.RESET}
  técnica · exploratoria · creativa · factual · razonamiento · conversacional
"""

# ── Colores por tipo de tarea ─────────────────────────────────────────────────
TASK_COLORS = {
    "técnica":        C.BLUE,
    "exploratoria":   C.CYAN,
    "creativa":       C.MAGENTA,
    "factual":        C.GREEN,
    "razonamiento":   C.YELLOW,
    "conversacional": C.WHITE,
}

TASK_ICONS = {
    "técnica":        "⚙️ ",
    "exploratoria":   "🔍",
    "creativa":       "✨",
    "factual":        "📋",
    "razonamiento":   "🧠",
    "conversacional": "💬",
}

# ── Carga del enjambre ────────────────────────────────────────────────────────

def load_swarm_for_chat(checkpoint: str = None, device: str = None):
    """Carga LixyOrchestrator con el mejor checkpoint disponible."""
    SRC_DIR = Path(__file__).parent
    sys.path.insert(0, str(SRC_DIR))

    from lixy_orchestrator import LixyOrchestrator, OrchestratorConfig

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = OrchestratorConfig(
        checkpoint=checkpoint or "auto",
        device=device,
        temperature=0.8,
        top_k=50,
        max_tokens=200,
        eval_matriarca=False,   # activar con --eval-matriarca
    )
    lixy = LixyOrchestrator(cfg)
    return lixy


# ── Formato de respuesta ──────────────────────────────────────────────────────

def format_response(response: str) -> str:
    """Limpia y formatea la respuesta del modelo."""
    # Remover tokens especiales que puedan filtrarse
    response = response.replace("<|endoftext|>", "").strip()

    # Si la respuesta está vacía, mensaje amable
    if not response:
        return f"{C.DIM}[sin respuesta]{C.RESET}"

    return response


def format_stats_line(turn_record, debug: bool = False) -> str:
    """Formatea la línea de stats por turno."""
    task = turn_record.task_type
    icon = TASK_ICONS.get(task, "·")
    color = TASK_COLORS.get(task, C.WHITE)

    stats = (
        f"{C.DIM}  {icon} {color}{task}{C.RESET}{C.DIM} · "
        f"{turn_record.n_tokens} tok"
    )

    if debug:
        stats += (
            f" · 🐘 infra={turn_record.infrasound_norm:.2f}"
            f" · 🐜 fero={turn_record.feromon_norm:.2f}"
        )

    stats += f"{C.RESET}"
    return stats


# ── Mini-eval de calidad ──────────────────────────────────────────────────────

def run_mini_eval(lixy, debug: bool = False):
    """
    Ejecuta 10 prompts variados y reporta calidad de generación.
    Detects: repeat ratio, task type, agente dominante.
    """
    prompts = [
        # técnica
        ("técnica",        "How do I implement attention in PyTorch?"),
        ("técnica",        "Explain the difference between TCP and UDP protocols"),
        # exploratoria
        ("exploratoria",   "What is consciousness and how does it emerge?"),
        ("exploratoria",   "Qué es el aprendizaje por refuerzo?"),
        # creativa
        ("creativa",       "Write a short poem about artificial intelligence"),
        ("creativa",       "Crea un haiku sobre la primavera"),
        # factual
        ("factual",        "Who invented the transformer architecture?"),
        ("factual",        "What year was Python first released?"),
        # razonamiento
        ("razonamiento",   "Why is batch normalization useful in deep learning?"),
        # conversacional
        ("conversacional", "Hola, ¿cómo estás hoy?"),
    ]

    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}🔬 Mini-eval de calidad — 10 prompts{C.RESET}")
    print(f"{'='*60}\n")

    results = []
    rs = lixy._runtime_session

    for expected_type, prompt in prompts:
        response = lixy.chat(prompt, store=False)   # no contaminar el historial real
        response_clean = response.replace("<|endoftext|>", "").strip()

        words = response_clean.split()
        n = len(words)
        unique = set(words)
        repeat_ratio = round(1.0 - len(unique) / max(1, n), 2)

        # Último turno = este
        last_turn = rs.history[-1] if rs.history else None
        detected_type = last_turn.task_type if last_turn else "?"

        icon = TASK_ICONS.get(detected_type, "·")
        color = TASK_COLORS.get(detected_type, C.WHITE)
        rep_color = C.RED if repeat_ratio > 0.5 else (C.YELLOW if repeat_ratio > 0.3 else C.GREEN)

        print(f"{C.DIM}Prompt:{C.RESET} {prompt[:55]}")
        print(f"  Esperado: {expected_type:15s} | Detectado: {color}{icon} {detected_type}{C.RESET}")
        print(f"  Respuesta: {repr(response_clean[:70])}")
        print(f"  Repetición: {rep_color}{repeat_ratio:.0%}{C.RESET} | Tokens: {last_turn.n_tokens if last_turn else 'N/A'}")
        print()

        results.append({
            "prompt": prompt,
            "expected": expected_type,
            "detected": detected_type,
            "correct": detected_type == expected_type,
            "repeat_ratio": repeat_ratio,
            "n_tokens": last_turn.n_tokens if last_turn else 0,
            "response_preview": response_clean[:80],
        })

    # Resumen
    correct = sum(r["correct"] for r in results)
    avg_repeat = sum(r["repeat_ratio"] for r in results) / len(results)
    avg_tokens = sum(r["n_tokens"] for r in results) / len(results)

    print(f"\n{C.BOLD}📊 Resumen:{C.RESET}")
    print(f"  Clasificación correcta: {C.GREEN}{correct}/{len(results)}{C.RESET}")
    print(f"  Repeat ratio promedio:  {C.GREEN if avg_repeat < 0.3 else C.RED}{avg_repeat:.1%}{C.RESET}")
    print(f"  Tokens promedio:        {avg_tokens:.1f}")
    print(f"  Distribución de tipos:  {rs._role_adapter.task_distribution()}")
    print()

    return results


# ── CLI Principal ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chat con Lixy-0.1 🐜🐘🐬")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt a usar")
    parser.add_argument("--no-color", action="store_true", help="Desactivar colores")
    parser.add_argument("--debug", action="store_true", help="Stats extendidos por turno")
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU")
    parser.add_argument("--eval", action="store_true", help="Ejecutar mini-eval y salir")
    args = parser.parse_args()

    if args.no_color:
        C.disable()

    device = "cpu" if args.cpu else None

    # ── Cargar modelo ──────────────────────────────────────────────────────────
    print(BANNER)
    print(f"{C.DIM}Cargando enjambre...{C.RESET}", flush=True)
    t0 = time.time()

    # Suprimir output de carga excepto errores críticos
    import io, contextlib
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            lixy = load_swarm_for_chat(args.checkpoint, device)
    except Exception as e:
        print(f"\n{C.RED}❌ Error al cargar el modelo: {e}{C.RESET}")
        sys.exit(1)

    load_time = time.time() - t0

    # Mostrar resumen de carga
    s = lixy.status()
    rs = lixy._runtime_session
    prev_turns = len(rs.history)

    print(f"\r{C.GREEN}✅ Enjambre listo{C.RESET} ({load_time:.1f}s)")
    print(f"  {C.DIM}Checkpoint:{C.RESET} step={lixy.swarm._last_step if hasattr(lixy.swarm, '_last_step') else '?'}")
    print(f"  {C.DIM}Matriarca:{C.RESET}  {s['matriarca_memories']} memorias")
    print(f"  {C.DIM}Agentes:{C.RESET}    {s['specialization_labels']}")
    if prev_turns > 0:
        print(f"  {C.DIM}Historial:{C.RESET}  {prev_turns} turnos anteriores restaurados")
    print()

    # ── Modo eval ─────────────────────────────────────────────────────────────
    if args.eval:
        run_mini_eval(lixy, debug=args.debug)
        lixy.close()
        return

    # ── Handler de señales ─────────────────────────────────────────────────────
    def graceful_exit(sig, frame):
        print(f"\n\n{C.YELLOW}👋 Hasta luego! Guardando sesión...{C.RESET}")
        lixy.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    # ── Loop principal ─────────────────────────────────────────────────────────
    print(f"{C.BOLD}Lixy está lista.{C.RESET} {C.DIM}Escribe /ayuda para ver comandos.{C.RESET}\n")

    turn_count = 0

    while True:
        # Prompt del usuario
        try:
            user_input = input(f"{C.GREEN}{C.BOLD}Tú:{C.RESET} ").strip()
        except EOFError:
            graceful_exit(None, None)

        if not user_input:
            continue

        # ── Comandos ──────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()

            if cmd in ("/salir", "/exit", "/quit"):
                graceful_exit(None, None)

            elif cmd in ("/ayuda", "/help"):
                print(HELP_TEXT)
                continue

            elif cmd == "/status":
                s = lixy.status()
                rs = lixy._runtime_session
                print(f"\n{C.BOLD}🐜🐘🐬 Estado del Enjambre{C.RESET}")
                print(f"  🐘 Matriarca:    {s['matriarca_memories']} memorias")
                print(f"  🐬 Delfín sleep: {s['dolphin_sleep_norm']:.4f}")
                print(f"  💬 Esta sesión:  {s['runtime_session']['turns']} turnos | {s['runtime_session']['tokens']} tokens")
                print(f"  🌡  Feromon:      {'activo' if s['runtime_session']['feromon_warm'] else 'frío'}")
                print(f"  🐜 Especialización: {s['specialization_labels']}")
                print()
                continue

            elif cmd == "/stats":
                rs = lixy._runtime_session
                dist = rs._role_adapter.task_distribution()
                print(f"\n{C.BOLD}📊 Tipos de tarea detectados (esta sesión):{C.RESET}")
                if dist:
                    for task, pct in dist.items():
                        icon = TASK_ICONS.get(task, "·")
                        color = TASK_COLORS.get(task, C.WHITE)
                        bar = "█" * int(pct * 20)
                        print(f"  {icon} {color}{task:15s}{C.RESET} {bar} {pct:.0%}")
                else:
                    print("  Sin datos aún")
                print()
                continue

            elif cmd == "/reset":
                lixy._runtime_session.reset_feromon()
                lixy.session_history = []
                print(f"{C.DIM}  ✓ Contexto reiniciado{C.RESET}\n")
                continue

            elif cmd == "/guardar":
                lixy._runtime_session._save_history()
                lixy._save_session()
                print(f"{C.DIM}  ✓ Sesión guardada{C.RESET}\n")
                continue

            elif cmd == "/eval":
                run_mini_eval(lixy, debug=args.debug)
                continue

            else:
                print(f"{C.DIM}  Comando desconocido. Escribe /ayuda.{C.RESET}\n")
                continue

        # ── Generar respuesta ─────────────────────────────────────────────────
        print(f"\n{C.CYAN}{C.BOLD}Lixy:{C.RESET} ", end="", flush=True)
        t_start = time.time()

        response = lixy.chat(user_input)
        response_clean = format_response(response)

        t_elapsed = time.time() - t_start

        print(response_clean)

        # Stats de turno
        rs = lixy._runtime_session
        last_turn = rs.history[-1] if rs.history else None
        if last_turn:
            stats_line = format_stats_line(last_turn, debug=args.debug)
            if args.debug:
                stats_line += f"{C.DIM} · {t_elapsed:.1f}s{C.RESET}"
            print(stats_line)

        print()
        turn_count += 1


if __name__ == "__main__":
    main()
