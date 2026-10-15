"""Desktop entry point; a frozen executable also runs its independent node."""
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    if '--memory-fixture-check' in sys.argv:
        from desktop.memory_fixture_check import main as memory_check
        raise SystemExit(memory_check())
    if "--node" in sys.argv:
        sys.argv.remove("--node")
        from client_daemon import main as node_main
        node_main()
        return
    if "--self-test" in sys.argv:
        from desktop.self_test import main as self_test
        raise SystemExit(self_test())
    from desktop.window import main as window_main
    window_main()


if __name__ == "__main__":
    main()
