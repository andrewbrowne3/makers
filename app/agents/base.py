"""Manual ReAct loop — mirrors Andrew's agent1.py style.

Agents output structured text:

  THINK: <reasoning>
  ACT: tool_name(arg1="..", arg2="..")
  OBSERVE: <tool result>    # filled in by the parent loop
  FINAL: <answer>

Parent loop regex-parses ACT lines, executes the tool, appends an
OBSERVE line, and continues until FINAL appears or max_steps is hit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.logging_config import get_logger
from app.providers.llm import LLMProvider

log = get_logger("agent.base")

ACT_RE = re.compile(r"^ACT:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)\s*$", re.MULTILINE)
FINAL_RE = re.compile(r"^FINAL:\s*(.+)$", re.MULTILINE | re.DOTALL)
KWARG_RE = re.compile(r"""(\w+)\s*=\s*(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)')""")


@dataclass
class Trace:
    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **kw: Any) -> None:
        self.steps.append(kw)


Tool = Callable[..., Any]


class ReactAgent:
    name: str = "react_agent"
    system_prompt: str = "You are a helpful ReAct agent. Use THINK / ACT / FINAL blocks."
    max_steps: int = 6
    min_tool_calls: int = 0  # reject FINAL until this many tools have been called

    def __init__(self, llm: LLMProvider, tools: dict[str, Tool]) -> None:
        self.llm = llm
        self.tools = tools

    def _tool_doc(self) -> str:
        lines = []
        for name, fn in self.tools.items():
            doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            lines.append(f"- {name}(...) — {doc}")
        return "\n".join(lines)

    def run(self, user_input: str, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        context = context or {}
        trace = Trace()
        system = (
            f"{self.system_prompt}\n\n"
            f"Available tools:\n{self._tool_doc()}\n\n"
            "Respond in this exact format:\n"
            "THINK: <your reasoning>\n"
            "ACT: tool_name(arg=\"value\", ...)\n"
            "— or —\n"
            "FINAL: <your final answer as JSON or plain text>\n"
            "Do NOT include OBSERVE: yourself; the system fills it in."
        )

        history: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]

        tool_calls_so_far = 0

        for step in range(1, self.max_steps + 1):
            log.info("🧭 %s step=%d", self.name, step)
            reply = self.llm.chat(history)
            log.info("🧭 %s raw_reply=%s", self.name, _truncate(reply, 400))
            history.append({"role": "assistant", "content": reply})

            final = FINAL_RE.search(reply)
            if final:
                if tool_calls_so_far < self.min_tool_calls:
                    log.warning(
                        "⚠️  %s premature FINAL (tools=%d < min=%d) — rejecting and nudging",
                        self.name, tool_calls_so_far, self.min_tool_calls,
                    )
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                f"You called FINAL without completing the required work. "
                                f"You have made {tool_calls_so_far} tool calls but need at least "
                                f"{self.min_tool_calls}. Do NOT respond with FINAL yet. "
                                f"Call the next ACT: in the required sequence."
                            ),
                        }
                    )
                    trace.add(step=step, kind="final_rejected", answer=final.group(1).strip())
                    continue
                answer = final.group(1).strip()
                trace.add(step=step, kind="final", answer=answer)
                log.info("✅ %s FINAL=%s", self.name, _truncate(answer, 200))
                return {"answer": answer, "trace": trace.steps, "steps": step}

            act = ACT_RE.search(reply)
            if not act:
                log.warning("⚠️  %s no ACT/FINAL in reply — nudging", self.name)
                history.append({"role": "user", "content": "Please respond with either ACT: or FINAL: as specified."})
                trace.add(step=step, kind="nudge")
                continue

            tool_name, raw_args = act.group(1), act.group(2)
            kwargs = {m.group(1): (m.group(2) or m.group(3) or "") for m in KWARG_RE.finditer(raw_args)}

            trace.add(step=step, kind="act", tool=tool_name, args=kwargs)
            log.info("🛠️  %s ACT %s args=%s", self.name, tool_name, kwargs)

            if tool_name not in self.tools:
                observation = f"ERROR: unknown tool {tool_name!r}. Available: {list(self.tools)}"
            else:
                try:
                    observation = str(self.tools[tool_name](**kwargs, _ctx=context))
                except TypeError:
                    try:
                        observation = str(self.tools[tool_name](**kwargs))
                    except Exception as e:  # noqa: BLE001
                        observation = f"ERROR calling {tool_name}: {e}"
                except Exception as e:  # noqa: BLE001
                    observation = f"ERROR calling {tool_name}: {e}"

            trace.add(step=step, kind="observe", result=_truncate(observation, 500))
            log.info("👁️  %s OBSERVE=%s", self.name, _truncate(observation, 200))
            history.append({"role": "user", "content": f"OBSERVE: {observation}"})
            if not observation.startswith("ERROR"):
                tool_calls_so_far += 1

        log.warning("⏱️  %s hit max_steps=%d without FINAL", self.name, self.max_steps)
        return {"answer": None, "trace": trace.steps, "steps": self.max_steps, "error": "max_steps_reached"}


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"
