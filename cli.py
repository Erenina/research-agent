"""
Komut satırı arayüzü — araştırma agent'ını çalıştır ve düşünme izini (trace) göster.

Kullanım:
    python cli.py "Sorunuz burada"
    python cli.py                 # etkileşimli mod (ardışık sorular)
"""

import sys

from agent.core import run_agent, AgentError


def _print_event(*ev):
    kind = ev[0]
    if kind == "think":
        print(f"\n💭 {ev[1]}")
    elif kind == "tool_call":
        name, args = ev[1], ev[2]
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"\n🔧 {name}({arg_str})")
    elif kind == "tool_result":
        result = str(ev[2]).strip().replace("\n", " ")
        preview = result if len(result) <= 300 else result[:300] + "…"
        print(f"   ↳ {preview}")
    elif kind == "final":
        print("\n" + "=" * 72)
        print("✅ CEVAP\n")
        print(ev[1])
        print("=" * 72)


def ask(question: str) -> None:
    print(f"\n❓ {question}")
    try:
        run_agent(question, on_event=_print_event)
    except AgentError as e:
        print(f"\n❌ {e}")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return
    print("🔎 Araştırma agent'ı hazır — çıkmak için Ctrl+C")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                ask(q)
    except (KeyboardInterrupt, EOFError):
        print("\ngörüşürüz 👋")


if __name__ == "__main__":
    main()
