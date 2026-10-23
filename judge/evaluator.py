"""LLM judge runner for the text face of one probe.

One probe, one judge, one file: `judge_output/<judge_id>/<probe_id>.json` under
the run directory. The runner reads only the *judge input* group of
`judge_input/<probe_id>/` (`meta.json`, `text_judge_pack.json`, `tool_log.jsonl`,
`ground_truth.json`, `scene.json`) plus `ix_selection.json` for the code-computed
`ix_face` block — the selection never reaches the prompt.

The prompt is seven blocks in this order, evidence before criteria:
① role + universal anchors (`prompts/system.md` + `prompts/anchors.md`),
② probe info, ③ scene, ④ transcript, ⑤ tool trajectory,
⑥ attribute rubric (`prompts/features/<attribute>.md`), ⑦ output schema
(`prompts/user.md`). Block ⑥ carries only the judge-facing half of the feature
file (everything above `## 备注（不进 judge prompt）`), and ③ / ⑤ shrink to a
one-line by-design note for the attributes whose criteria do not use them
(`UNUSED_BLOCKS`); every block keeps its heading and its place in the order.

Provider/model are parameters; every call goes through OpenRouter with the same
streaming wall-clock-deadline pattern as the authoring generator.
`judge_id` is `llm__<provider>_<model>`.

Run (no provider call):
    PYTHONPATH=$(pwd) python -m judge.evaluator \\
        --judge-input <run>/judge_input/fin_02 --model google/gemini-3.8-flash --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import httpx

from data.personas.coverage import TAXONOMY
from harness.judge_pack import text_judge_tool_log
from judge.evidence import check_text_face_evidence
from judge.ix_match import ix_face

# The OpenRouter helpers of the authoring generator, copied verbatim.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Models whose JSON mode returns an empty object on this prompt's length.
_NO_JSON_MODE_PREFIXES = ("deepseek/", "z-ai/")
_SUPPORTED_PARAMETERS: dict[str, frozenset[str]] = {}


def _supported_parameters(model: str, headers: dict[str, str]) -> frozenset[str]:
    """OpenRouter's declared parameter support for one model, fetched once per process."""
    if model not in _SUPPORTED_PARAMETERS:
        models_url = OPENROUTER_URL.rsplit("/chat/", 1)[0] + "/models"
        # A transient connect timeout here must not cost a whole generation attempt.
        for preflight in range(3):
            try:
                response = httpx.get(models_url, headers=headers, timeout=60.0)
                response.raise_for_status()
                break
            except httpx.HTTPError as exc:
                print(f"openrouter_preflight_error attempt={preflight + 1}/3 error={exc}", file=sys.stderr, flush=True)
                if preflight == 2:
                    raise
                time.sleep(5)
        entries = {
            item["id"]: item
            for item in response.json().get("data", [])
            if isinstance(item, dict)
        }
        if model not in entries:
            raise RuntimeError(f"OpenRouter does not list model {model!r}")
        _SUPPORTED_PARAMETERS[model] = frozenset(
            entries[model].get("supported_parameters") or ()
        )
    return _SUPPORTED_PARAMETERS[model]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FEATURES_DIR = PROMPTS_DIR / "features"
JUDGE_OUTPUT_SCHEMA_VERSION = 1
JUDGE_KIND = "llm"
SCORE_DOMAIN = [1, 2, 3, 4, 5]
#: Files of the judge-input group the text face reads.
TEXT_FACE_INPUT_FILES = (
    "meta.json",
    "text_judge_pack.json",
    "tool_log.jsonl",
    "ground_truth.json",
    "scene.json",
)
#: Everything below this heading in a feature file is human commentary (rubric
#: provenance, literature line references, open design questions) and stays out of
#: block ⑥. A feature file without the heading is used whole.
COMMENTARY_HEADING = "## 备注（不进 judge prompt）"
#: The two prompt blocks an attribute's row can rule out.
BLOCK_SCENE = "scene"
BLOCK_TOOL_TRAJECTORY = "tool_trajectory"
#: Column 「用不到的文件」 (files an attribute does not use), one entry per row (the
#: comment quotes that row's cell). `state_diff` / `pa_toolcalls` are named in that
#: column too but are not prompt blocks, so they leave nothing here; ② probe info,
#: ④ transcript, ⑥ rubric and ⑦ output are never dropped. A row that rules a block
#: out only in part keeps it — `guidance_level` (scene 只用 `familiarity_signals`)
#: and `capability_boundary` (only `scene.triggers[]` 明确不参与) both keep `scene`.
UNUSED_BLOCKS: dict[str, frozenset[str]] = {
    # tool_log、state_diff、pa_toolcalls、scene
    "tone_formality": frozenset({BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY}),
    # tool_log、state_diff、pa_toolcalls、scene
    "verbosity": frozenset({BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY}),
    # tool_log、state_diff、pa_toolcalls、scene
    "emotional_engagement": frozenset({BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY}),
    # tool_log、state_diff、pa_toolcalls；scene 只用 `familiarity_signals` → scene 保留
    "guidance_level": frozenset({BLOCK_TOOL_TRAJECTORY}),
    # tool_log、state_diff、pa_toolcalls、scene
    "reasoning_visibility": frozenset({BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY}),
    # tool_log、state_diff、pa_toolcalls、scene
    "uncertainty_expression": frozenset({BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY}),
    # state_diff、scene
    "process_visibility": frozenset({BLOCK_SCENE}),
    # state_diff、scene
    "autonomy_level": frozenset({BLOCK_SCENE}),
    # state_diff；`pa_toolcalls.json` 不可用（tool_log 判越界、scene 未列为用不到）
    "proactive_outreach": frozenset(),
    # pa_toolcalls（仅备查）
    "task_expansion": frozenset(),
    # state_diff、pa_toolcalls
    "solution_breadth": frozenset(),
    # state_diff、pa_toolcalls；`scene.triggers[]` 明确不参与 → scene 保留
    "capability_boundary": frozenset(),
    # state_diff；scene 保留 —— judge 要拿到 `withheld_required_facts` 清单判
    # `Structured` 问没问全
    "information_elicitation": frozenset(),
    # state_diff、pa_toolcalls、scene
    "topic_management": frozenset({BLOCK_SCENE}),
}
#: What a dropped block says instead of its content, so the judge reads an
#: intentional omission rather than a missing section.
OMITTED_NOTE = {
    BLOCK_SCENE: "（按设计省略：本属性的判据用不到场景块——见设计 §4.4「用不到的文件」一列。）",
    BLOCK_TOOL_TRAJECTORY: "（按设计省略：本属性的判据用不到工具轨迹——见设计 §4.4「用不到的文件」一列。）",
}
#: The only keys the model may return; anything else is a schema violation.
REPLY_KEYS = frozenset({"score", "rationale", "evidence", "asked_user_for_preference", "flags"})
FLAG_KEYS = frozenset({"content_error_corrected_by_simulator", "task_failed", "probe_degenerate"})
#: Plenty for a JSON verdict; nowhere near the 65536 the generators reserve.
JUDGE_MAX_TOKENS = 8192
DEFAULT_DEADLINE_SECONDS = 300
_ARGS_SUMMARY_CHARS = 200


class JudgeOutputInvalid(RuntimeError):
    """The model's reply failed the schema or the evidence check twice."""


# --------------------------------------------------------------------------- input


@dataclass(frozen=True)
class JudgeInput:
    probe_dir: Path
    meta: dict[str, Any]
    text_pack: dict[str, Any]
    tool_log: list[dict[str, Any]]
    ground_truth: dict[str, Any]
    scene: dict[str, Any]
    ix_selection: dict[str, Any]

    @property
    def probe_id(self) -> str:
        return self.probe_dir.name

    @property
    def attribute(self) -> str:
        return self.ground_truth["attribute"]

    @property
    def pa_outbounds(self) -> list[str]:
        return [
            turn["public_turn"]
            for turn in self.text_pack["turns"]
            if turn.get("kind") == "pa_turn"
        ]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_judge_input(probe_dir: Path) -> JudgeInput:
    """Read `judge_input/<probe_id>/`."""
    probe_dir = Path(probe_dir)
    return JudgeInput(
        probe_dir=probe_dir,
        meta=_read_json(probe_dir / "meta.json"),
        text_pack=_read_json(probe_dir / "text_judge_pack.json"),
        tool_log=_read_jsonl(probe_dir / "tool_log.jsonl"),
        ground_truth=_read_json(probe_dir / "ground_truth.json"),
        scene=_read_json(probe_dir / "scene.json"),
        ix_selection=_read_json(probe_dir / "ix_selection.json"),
    )


# -------------------------------------------------------------------------- prompt


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def rubric_markdown(attribute: str) -> str:
    """Block ⑥ for one attribute, plus the anchors — also what `rubric.md` holds.

    Only the judge-facing half of the feature file: definition, feature table and
    the 「不算本属性的情况 / 任务失败」 clauses. Everything from
    `COMMENTARY_HEADING` down is human commentary and never reaches the prompt.
    """
    if attribute not in TAXONOMY:
        raise KeyError(f"unknown attribute {attribute!r}; expected one of {sorted(TAXONOMY)}")
    feature = (FEATURES_DIR / f"{attribute}.md").read_text(encoding="utf-8")
    feature = feature.split(COMMENTARY_HEADING, 1)[0]
    settings = ", ".join(f"`{s}`" for s in TAXONOMY[attribute])
    return f"**属性** `{attribute}` — 现行 setting 列表：{settings}\n\n{feature.rstrip()}\n"


def _probe_info(inp: JudgeInput) -> str:
    gt = inp.ground_truth
    evolving = gt["evolving"]
    lines = [
        f"- probe_id: `{inp.probe_id}`",
        f"- probe_kind: `{inp.meta['kind']}`",
        f"- context: `{gt['context']}`",
        f"- 目标属性: `{gt['attribute']}`",
        f"- ground-truth setting: **`{gt['setting']}`**",
        f"- 该属性的选择域: {', '.join(f'`{s}`' for s in gt['allowed_settings'])}",
    ]
    if evolving["is_evolving"]:
        lines.append(
            "- evolving cell: 是 —— "
            f"pre_setting `{evolving['pre_setting']}` → post_setting `{evolving['post_setting']}`；"
            "pre_event probe 对 pre_setting 打分，final probe 对 post_setting 打分；"
            "上面的 ground-truth setting 已经是本条 probe 该对照的那一个"
        )
    else:
        lines.append("- evolving cell: 否")
    return "\n".join(lines)


def _scene(inp: JudgeInput) -> str:
    scene = inp.scene
    lines = [f"**核心任务**（用户明确要求的事）：{scene['core_task']}", "", "**世界触发物**："]
    triggers = scene.get("triggers") or []
    if not triggers:
        lines.append("（无）")
    for trigger in triggers:
        lines.append(
            f"- `{trigger['trigger_id']}` — background: {trigger['background']}"
            f" / detail: {trigger['detail']} / user_impact: {trigger['user_impact']}"
        )
    world = scene["world_state"]
    lines += ["", "**世界状态**（分支起点、PA 第一次调用之前）："]
    lines.append(f"- 可行选项 feasible_options（{len(world['feasible_options'])} 项）：")
    lines += [f"  - {json.dumps(item, ensure_ascii=False)}" for item in world["feasible_options"]] or ["  （无）"]
    lines.append(f"- 待处理事项 pending_items（{len(world['pending_items'])} 项）：")
    lines += [f"  - {json.dumps(item, ensure_ascii=False)}" for item in world["pending_items"]] or ["  （无）"]
    # 作者故意从
    # 用户请求里抽掉的必需事实清单只进 `information_elicitation` 的 judge prompt（`Structured`
    # 覆盖度的对照物）；别的带场景块的属性不渲染它。被测 PA 拿不到 scene.json。
    if inp.attribute == "information_elicitation":
        withheld = scene.get("withheld_required_facts") or []
        lines += ["", f"**作者扣住的必需事实** withheld_required_facts（{len(withheld)} 项；每条 = task_fact 名 + 作者说明）："]
        lines += [f"- `{fact.get('task_fact')}` — {fact.get('note')}" for fact in withheld] or ["（无）"]
    signals = scene.get("familiarity_signals")
    if signals:
        lines += [
            "",
            f"**术语对**（guidance_level 用）：用户会用的 known_term = `{signals['known_term']}`；"
            f"用户不懂的 unknown_term = `{signals['unknown_term']}`",
        ]
    return "\n".join(lines)


def _transcript(inp: JudgeInput) -> str:
    lines: list[str] = []
    for number, turn in enumerate(inp.text_pack["turns"], start=1):
        if turn["kind"] == "user_message":
            lines.append(f"[{number} · 用户 · t={turn['t']}] {turn['text']}")
        elif turn["kind"] == "environment_trigger":
            # 主动联系例外：联系机会（PA 收到的触发事件）。
            lines.append(
                f"[{number} · 联系机会 · t={turn['t']}] trigger_id=`{turn.get('trigger_id')}` "
                f"source_type={turn.get('source_type')} / prior_thread: {turn.get('prior_thread')}"
                f" / detail: {turn.get('detail')} / user_impact: {turn.get('user_impact')}"
            )
        else:
            if turn.get("proactive_event"):
                # 决定、投递情况、投递内容——只在 proactive 轮。
                delivered = bool(turn.get("outreach_delivered"))
                lines.append(
                    f"[{number} · PA 主动联系轮 · t={turn['t']}] proactive_event=true "
                    f"outreach_decision={turn.get('outreach_decision')} "
                    f"outreach_delivered={'true' if delivered else 'false'}；"
                    f"投递内容：{turn['public_turn'] if delivered else '（未投递，用户这一轮什么都没收到）'}"
                )
            else:
                lines.append(f"[{number} · PA · t={turn['t']}] {turn['public_turn']}")
            for event in turn.get("task_tool_events") or []:
                lines.append(
                    f"    ↳ 工具 {event['name']} status={event['status']} "
                    f"args={_summarize_args(event.get('args') or {})}"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


def _summarize_args(args: Mapping[str, Any]) -> str:
    text = json.dumps(args, ensure_ascii=False, sort_keys=True)
    if len(text) > _ARGS_SUMMARY_CHARS:
        text = text[:_ARGS_SUMMARY_CHARS] + "…"
    return text


def _tool_trajectory(inp: JudgeInput) -> str:
    # 业务结果与错误信息过滤后给——行、`result_summary` 走 `harness/judge_pack.py`
    # 的 allow-list，闸门 / 扣留 / 选档一类的控制字段不进来。
    lead = "按时间排序的世界工具调用（`t` 为轨迹序号；`result` 为该调用的业务结果摘要）。"
    rows = [
        f"- t={entry['t']} tool=`{entry['tool']}` status={entry.get('status', 'ok')} "
        f"args={_summarize_args(entry['args'])} result={_summarize_args(entry['result_summary'])}"
        for entry in text_judge_tool_log(inp.tool_log)
    ]
    return lead + "\n\n" + ("\n".join(rows) if rows else "（本条 probe 没有世界工具调用）")


OUTPUT_SCHEMA = """```json
{
  "score": <int，1–5 之一>,
  "rationale": "<string，说明为什么是这一档；指向特征表里的哪几条>",
  "evidence": [
    {"type": "quote", "text": "<PA 回复的逐字子串>"},
    {"type": "tool_event", "t": <int>, "tool": "<工具名>"}
  ],
  "asked_user_for_preference": <bool，PA 是否直接向用户索要了这项偏好>,
  "flags": {
    "content_error_corrected_by_simulator": <bool，用户在对话里纠正了 PA 的事实错误>,
    "task_failed": <bool，用户要求的事没有做成；可恢复的参数重调不算>,
    "probe_degenerate": <bool，这条 probe 的材料让本属性根本无从判断>
  }
}
```
`evidence` 至少一条，两种类型可混用。"""


def omitted_blocks(attribute: str) -> tuple[str, ...]:
    """The prompt blocks ruled out for this attribute, in prompt order."""
    unused = UNUSED_BLOCKS[attribute]
    return tuple(block for block in (BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY) if block in unused)


def assemble_prompt(inp: JudgeInput) -> tuple[str, str]:
    """(system, user) — the seven prompt blocks in order.

    A block ruled out for this attribute keeps its heading and carries
    `OMITTED_NOTE` instead of its content, so the order stays the same for every
    attribute and a dropped block never reads as a missing one.
    """
    dropped = omitted_blocks(inp.attribute)
    system = _load_prompt("system.md").replace("{{ANCHORS}}", _load_prompt("anchors.md").rstrip())
    user = (
        _load_prompt("user.md")
        .replace("{{PROBE_INFO}}", _probe_info(inp))
        .replace(
            "{{SCENE}}",
            OMITTED_NOTE[BLOCK_SCENE] if BLOCK_SCENE in dropped else _scene(inp),
        )
        .replace("{{TRANSCRIPT}}", _transcript(inp))
        .replace(
            "{{TOOL_TRAJECTORY}}",
            OMITTED_NOTE[BLOCK_TOOL_TRAJECTORY]
            if BLOCK_TOOL_TRAJECTORY in dropped
            else _tool_trajectory(inp),
        )
        .replace("{{RUBRIC}}", rubric_markdown(inp.attribute).rstrip())
        .replace("{{OUTPUT_SCHEMA}}", OUTPUT_SCHEMA)
    )
    return system, user


# --------------------------------------------------------------------------- reply


def parse_reply(text: str) -> dict[str, Any]:
    """The model's JSON, checked against the strict text-face shape.

    Raises `JudgeOutputInvalid` with the first violation found.
    """
    try:
        reply = json.loads(_strip_fences(text))
    except json.JSONDecodeError as exc:
        raise JudgeOutputInvalid(f"reply is not JSON: {exc}") from exc
    if not isinstance(reply, dict):
        raise JudgeOutputInvalid("reply is not a JSON object")
    if set(reply) != REPLY_KEYS:
        raise JudgeOutputInvalid(
            f"reply keys {sorted(reply)} != expected {sorted(REPLY_KEYS)}"
        )
    score = reply["score"]
    if isinstance(score, bool) or score not in SCORE_DOMAIN:
        raise JudgeOutputInvalid(f"score {score!r} not in {SCORE_DOMAIN}")
    if not isinstance(reply["rationale"], str) or not reply["rationale"].strip():
        raise JudgeOutputInvalid("rationale must be a non-empty string")
    evidence = reply["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise JudgeOutputInvalid("evidence must be a non-empty list")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise JudgeOutputInvalid(f"evidence[{index}] is not an object")
        if item.get("type") == "quote":
            if set(item) != {"type", "text"} or not isinstance(item["text"], str):
                raise JudgeOutputInvalid(f"evidence[{index}]: quote needs exactly type+text")
        elif item.get("type") == "tool_event":
            if (
                set(item) != {"type", "t", "tool"}
                or isinstance(item["t"], bool)
                or not isinstance(item["t"], int)
                or not isinstance(item["tool"], str)
            ):
                raise JudgeOutputInvalid(f"evidence[{index}]: tool_event needs exactly type+t(int)+tool")
        else:
            raise JudgeOutputInvalid(f"evidence[{index}]: type must be quote or tool_event")
    if not isinstance(reply["asked_user_for_preference"], bool):
        raise JudgeOutputInvalid("asked_user_for_preference must be a bool")
    flags = reply["flags"]
    if not isinstance(flags, dict) or set(flags) != FLAG_KEYS:
        raise JudgeOutputInvalid(f"flags keys must be exactly {sorted(FLAG_KEYS)}")
    if any(not isinstance(value, bool) for value in flags.values()):
        raise JudgeOutputInvalid("every flag must be a bool")
    return reply


# -------------------------------------------------------------------------- client


@dataclass(frozen=True)
class JudgeReply:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    usd: float | None


class JudgeClient(Protocol):
    def complete(self, *, system: str, user: str) -> JudgeReply: ...


def split_model_id(model: str) -> tuple[str, str]:
    """`google/gemini-3.8-flash` → (`google`, `gemini-3.8-flash`)."""
    provider, sep, name = model.partition("/")
    if not sep or not provider or not name or "/" in name:
        raise ValueError(f"model must be an OpenRouter id `<provider>/<model>`, got {model!r}")
    return provider, name


def judge_id_for(model: str) -> str:
    provider, name = split_model_id(model)
    return f"llm__{provider}_{name}"


class OpenRouterJudgeClient:
    """One OpenRouter chat completion per `complete`, with a wall-clock deadline.

    Same shape as the authoring generator's OpenRouter call: streamed
    body so keep-alive bytes cannot silently extend the request past
    `deadline_seconds`, JSON mode only where the model declares it, one
    unbuffered log line per call with tokens / seconds / USD.
    """

    def __init__(self, model: str, *, deadline_seconds: int = DEFAULT_DEADLINE_SECONDS) -> None:
        split_model_id(model)
        self.model = model
        self.deadline_seconds = deadline_seconds
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mempa-bench",
            "X-Title": "MemPABench Judge",
        }

    def complete(self, *, system: str, user: str) -> JudgeReply:
        return self.complete_payload(self.request_payload(system=system, user=user))

    def request_payload(self, *, system: str, user: str) -> dict[str, Any]:
        """Build the original judge request so an owner can persist its intent."""
        supported = _supported_parameters(self.model, self._headers)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": JUDGE_MAX_TOKENS,
            "usage": {"include": True},
        }
        if "response_format" in supported and not self.model.startswith(_NO_JSON_MODE_PREFIXES):
            payload["response_format"] = {"type": "json_object"}
            payload["provider"] = {"require_parameters": True}
        if "temperature" in supported:
            payload["temperature"] = 0.0
        return payload

    def complete_payload(self, payload: dict[str, Any]) -> JudgeReply:
        """Send exactly one stored request, without transport retry or re-ask."""
        if payload.get("model") != self.model or payload.get("max_tokens") != JUDGE_MAX_TOKENS:
            raise ValueError("Stored judge request changed model or completion limit")
        started = time.monotonic()
        print(
            f"judge_request model={self.model} prompt_chars={sum(len(message['content']) for message in payload['messages'])} "
            f"deadline_s={self.deadline_seconds}",
            file=sys.stderr,
            flush=True,
        )
        # HTTPX's default transport has no retry; keep the original proxy/env policy.
        with httpx.Client(timeout=self.deadline_seconds + 60) as client:
            with client.stream("POST", OPENROUTER_URL, headers=self._headers, json=payload) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    if time.monotonic() - started > self.deadline_seconds:
                        raise TimeoutError(
                            f"OpenRouter judge request exceeded {self.deadline_seconds}s deadline"
                        )
        body = json.loads(b"".join(chunks))
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        seconds = time.monotonic() - started
        print(
            f"judge_call model={body.get('model', self.model)} seconds={seconds:.1f} "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} cost_usd={usage.get('cost')} "
            f"finish={choice.get('finish_reason')}",
            file=sys.stderr,
            flush=True,
        )
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise JudgeOutputInvalid("response content is not text")
        if choice.get("finish_reason") == "length":
            raise JudgeOutputInvalid("response was truncated (finish_reason=length)")
        return JudgeReply(
            content=content,
            model=body.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            usd=usage.get("cost"),
        )


# ---------------------------------------------------------------------------- run


def _reask_user(user: str, previous: str, reasons: list[str]) -> str:
    return (
        f"{user}\n\n# 上一次输出被拒绝\n\n"
        "你上一次的输出没有通过机器校验，原因：\n"
        + "\n".join(f"- {reason}" for reason in reasons)
        + "\n\n上一次的输出：\n```\n"
        + previous
        + "\n```\n\n请重新给出完整的 JSON；`quote` 必须逐字复制 PA 回复，`tool_event` 必须指向工具轨迹里存在的 `t`。"
    )


def judge_probe(inp: JudgeInput, client: JudgeClient, *, model: str) -> dict[str, Any]:
    """Ask once, re-ask once on a schema or evidence failure, else raise.

    Returns the judge record; `cost` sums both calls when a re-ask happened.
    """
    system, user = assemble_prompt(inp)
    cost = {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    prompt = user
    verdict: dict[str, Any] | None = None
    for attempt in (1, 2):
        reply = client.complete(system=system, user=prompt)
        cost["model"] = reply.model
        cost["prompt_tokens"] += reply.prompt_tokens or 0
        cost["completion_tokens"] += reply.completion_tokens or 0
        cost["usd"] += reply.usd or 0.0
        try:
            candidate = parse_reply(reply.content)
        except JudgeOutputInvalid as exc:
            reasons = [str(exc)]
        else:
            reasons = check_text_face_evidence(
                candidate["evidence"], outbounds=inp.pa_outbounds, tool_log=inp.tool_log
            )
            if not reasons:
                verdict = candidate
                break
        print(
            f"judge_reject probe={inp.probe_id} attempt={attempt}/2 reasons={reasons}",
            file=sys.stderr,
            flush=True,
        )
        if attempt == 2:
            raise JudgeOutputInvalid(
                f"{inp.probe_id}: judge output rejected twice: " + "; ".join(reasons)
            )
        prompt = _reask_user(user, reply.content, reasons)
    assert verdict is not None
    return {
        "schema_version": JUDGE_OUTPUT_SCHEMA_VERSION,
        "probe_id": inp.probe_id,
        "judge_id": judge_id_for(model),
        "judge_kind": JUDGE_KIND,
        "run_id": inp.meta["run_id"],
        "ix_face": ix_face(inp.ix_selection, inp.ground_truth),
        "text_face": {
            "score": verdict["score"],
            "score_domain": list(SCORE_DOMAIN),
            "rationale": verdict["rationale"],
            "evidence": verdict["evidence"],
            "asked_user_for_preference": verdict["asked_user_for_preference"],
        },
        "flags": verdict["flags"],
        "cost": cost,
    }


def judge_output_path(run_dir: Path, judge_id: str, probe_id: str) -> Path:
    return Path(run_dir) / "judge_output" / judge_id / f"{probe_id}.json"


def run_judge(probe_dir: Path, client: JudgeClient, *, model: str, run_dir: Path | None = None) -> Path:
    """Judge one `judge_input/<probe_id>/` and write its judge record."""
    probe_dir = Path(probe_dir)
    inp = load_judge_input(probe_dir)
    record = judge_probe(inp, client, model=model)
    out = judge_output_path(run_dir or probe_dir.parent.parent, record["judge_id"], inp.probe_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"judge_done probe={inp.probe_id} judge_id={record['judge_id']} "
        f"score={record['text_face']['score']} usd={record['cost']['usd']} out={out}",
        file=sys.stderr,
        flush=True,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--judge-input",
        required=True,
        type=Path,
        help="Path to one probe's judge input directory: `<run_dir>/judge_input/<probe_id>` "
        "(probe_id matches `^(fin|pre)_\\d{2}$`).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model id `<provider>/<model>`, e.g. `google/gemini-3.8-flash`; "
        "judge_id becomes `llm__<provider>_<model>`.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory that receives `judge_output/<judge_id>/<probe_id>.json`; "
        "default: two levels above --judge-input.",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=int,
        default=DEFAULT_DEADLINE_SECONDS,
        help=f"Wall-clock cap per OpenRouter request in seconds (positive int; default {DEFAULT_DEADLINE_SECONDS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assembled system and user prompt to stdout and call nothing.",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        inp = load_judge_input(args.judge_input)
        system, user = assemble_prompt(inp)
        dropped = ", ".join(omitted_blocks(inp.attribute)) or "（无）"
        print(
            f"### judge_id: {judge_id_for(args.model)}\n"
            f"### attribute: {inp.attribute}\n"
            f"### omitted blocks (§4.4): {dropped}\n"
            f"### prompt_chars: {len(system) + len(user)}\n"
            f"\n### SYSTEM\n\n{system}\n\n### USER\n\n{user}"
        )
        return 0
    client = OpenRouterJudgeClient(args.model, deadline_seconds=args.deadline_seconds)
    run_judge(args.judge_input, client, model=args.model, run_dir=args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
