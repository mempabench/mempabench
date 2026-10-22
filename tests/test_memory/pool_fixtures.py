"""One realistic turn pool, shared by every adapter's step-6 test.

The pool is built by `memory.event_writer.build_turn_pool` — never hand-rolled
here — so the adapter tests assert against the bytes the harness really hands
them.  It carries dialogue plus `[reasoning]`, `[prose]`, `[tool_call]`,
`[ix_selection]` and `[tool_result]` blocks.
"""

from __future__ import annotations

from datetime import datetime, timezone

from memory.event_writer import TurnPool, build_turn_pool
from memory.events import MemoryEvent


#: One turn's raw provider-loop messages, in the PA's own order.
LOOP_MESSAGES: list[dict[str, object]] = [
    {"role": "user", "content": "Book me a table for Thursday."},
    {
        "role": "assistant",
        "content": "Let me check the calendar first.",
        "reasoning_content": "Thursday is ambiguous; read the calendar.",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "calendar_read",
                    "arguments": '{"date":"2026-03-05"}',
                },
            },
            {
                "id": "call_2",
                "function": {
                    "name": "IX_verbosity",
                    "arguments": '{"evidence":"one-line replies","setting":"Concise"}',
                },
            },
        ],
    },
    {
        "role": "tool",
        "name": "calendar_read",
        "tool_call_id": "call_1",
        "content": '{"events":[]}',
    },
    {
        "role": "assistant",
        "content": "Thursday 7pm is free — booked.",
        "reasoning_content": "Nothing conflicts; confirm briefly.",
    },
]


def full_experience_pool() -> TurnPool:
    return build_turn_pool(LOOP_MESSAGES)


def pool_event(pool: TurnPool | None = None) -> MemoryEvent:
    turn_pool = pool if pool is not None else full_experience_pool()
    return MemoryEvent(
        event_id="evt_pool_1",
        run_id="run_001",
        persona_id="user_a",
        agent_id="nanobot",
        session_id="s01",
        turn_id="t01",
        timestamp=datetime(2026, 3, 3, 9, 0, tzinfo=timezone.utc),
        messages=list(turn_pool.messages),
        items=list(turn_pool.items),
    )


def expected_contents(pool: TurnPool) -> list[str]:
    return [message["content"] for message in pool.messages]


def acc_sized_turn_pool() -> TurnPool:
    """A turn pool the size the live ACC line really produces.

    Shaped from the run that failed: 13 pool messages, two oversized tool
    results (10 371 and 16 111 characters live), the rest reasoning, tool
    calls and dialogue. The bodies here are realistic JSON rather than filler
    so they tokenize at the ~4.2 characters/token the live payload did; a
    random-character body would tokenize at half that and understate the
    problem.
    """
    paper = (
        '{"id":"arXiv:2411.%05d","title":"Environment-induced superselection '
        'in open quantum systems","authors":["M. Brand\\u00e3o","R. Oliveira"],'
        '"year":2024,"abstract":"We analyse einselection under non-Markovian '
        'noise and derive the pointer-state basis for a damped oscillator.",'
        '"keywords":["decoherence","einselection","pointer states"]}'
    )
    search_result = (
        '{"query":"einselection pointer states","results":['
        + ",".join(paper % n for n in range(55))
        + "]}"
    )
    notes = (
        '{"path":"documents/research/citation_notes.md","content":'
        + '"## Brand\\u00e3o 2024 — einselection\\nThe pointer basis is fixed '
        'by the interaction Hamiltonian, not by the system Hamiltonian. '
        'Compare against Zurek 1981 for the original argument.\\n"'
        * 70
        + "}"
    )
    draft = (
        '{"path":"documents/drafts/chapter_03.md","content":'
        + '"The pointer basis argument carries over to the damped case, but '
        'the decoherence time scales differently; see the note beside '
        'equation (14).\\n"'
        * 64
        + "}"
    )
    loop_messages: list[dict[str, object]] = [
        {"role": "user", "content": "Pull up my notes on the Brandao paper."},
        {
            "role": "assistant",
            "content": "Let me read the research folder first.",
            "reasoning_content": (
                "The user wants their own notes, so read the folder before "
                "searching anything external."
            ),
            "tool_calls": [
                {
                    "id": "call_ix",
                    "function": {
                        "name": "IX_verbosity",
                        "arguments": '{"evidence":"short replies","setting":"Terse"}',
                    },
                },
                {
                    "id": "call_read",
                    "function": {
                        "name": "mcp_state_document_read",
                        "arguments": '{"path":"documents/research"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "name": "IX_verbosity",
            "tool_call_id": "call_ix",
            "content": '{"type":"ix_preference_selection","setting":"Terse"}',
        },
        {
            "role": "tool",
            "name": "mcp_state_document_read",
            "tool_call_id": "call_read",
            "content": notes,
        },
        {
            "role": "assistant",
            "content": "Now let me check the literature for the exact citation.",
            "reasoning_content": "The notes name the paper but not the arXiv id.",
            "tool_calls": [
                {
                    "id": "call_search",
                    "function": {
                        "name": "mcp_state_scholarly_paper_search",
                        "arguments": '{"query":"Brandao einselection"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "mcp_state_scholarly_paper_search",
            "tool_call_id": "call_search",
            "content": search_result,
        },
        {
            "role": "assistant",
            "content": "Let me also check where the draft cites it.",
            "reasoning_content": "The user keeps citations in the chapter drafts.",
            "tool_calls": [
                {
                    "id": "call_draft",
                    "function": {
                        "name": "mcp_state_document_read",
                        "arguments": '{"path":"documents/drafts/chapter_03.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "mcp_state_document_read",
            "tool_call_id": "call_draft",
            "content": draft,
        },
        {
            "role": "assistant",
            "content": "arXiv:2411.00003 — pointer states under non-Markovian noise.",
            "reasoning_content": "Answer terse, as selected.",
        },
    ]
    return build_turn_pool(loop_messages)


#: One `reasoning_content` paragraph, at the ~4.3 characters/token the live
#: `z-ai/glm-5.3-flash` reasoning tokenized at. Repeated to reach a size, so
#: the fixture's token count comes from real prose rather than filler.
_REASONING_PARAGRAPH = (
    "She asked for the bar stock sheet, and the note she dictated has to go "
    "in verbatim: two cases of lemons for the weekend brunch service. Before "
    "I write anything I should check whether last week's sheet already "
    "carries a lemon line, because duplicating it would leave the kitchen "
    "ordering twice. The calendar says the brunch shift starts at ten, so "
    "the order has to be with the supplier tonight. I will read the existing "
    "sheet first, then the supplier note, then decide whether to ask her or "
    "just file it the way she has had me file the last four. "
)


def oversized_reasoning_turn_pool() -> TurnPool:
    """A pool whose ONE assistant message does not fit the embedding input.

    The shape that killed `ub_simple_rag_glm_5_3_flash_20260918` at
    `acc_014:beat_clarify_task:t1`: a single iteration whose reasoning block
    alone runs over the cap, followed by its tool calls, which
    `build_turn_pool` emits as one assistant message. The `[reasoning]` block
    therefore does not fit even on its own seam, so the cut has to land inside
    it.
    """
    loop_messages: list[dict[str, object]] = [
        {"role": "user", "content": "Can you add a line to the bar sheet for me?"},
        {
            "role": "assistant",
            "content": "Reading the sheet first.",
            "reasoning_content": _REASONING_PARAGRAPH * 75,
            "tool_calls": [
                {
                    "id": "call_sheet",
                    "function": {
                        "name": "mcp_state_document_read",
                        "arguments": '{"path":"documents/work/bar_stock_sheet.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "mcp_state_document_read",
            "tool_call_id": "call_sheet",
            "content": '{"path":"documents/work/bar_stock_sheet.md","content":"- limes x2\\n"}',
        },
        {
            "role": "assistant",
            "content": "Added two cases of lemons for the weekend brunch service.",
            "reasoning_content": "She dictated it; file it the way she files these.",
        },
    ]
    return build_turn_pool(loop_messages)


def oversized_selection_turn_pool() -> TurnPool:
    """A pool whose one oversized message is a seam away from fitting.

    The shape of `acc_017:request_dance_spot:t0`: a reasoning block that fits
    on its own, followed by forty-odd `[ix_selection]` blocks the wrapper
    deduplicated, all in one assistant message. Every block here fits, so the
    cut belongs on a block seam and nothing needs cutting inside a block.
    """
    loop_messages: list[dict[str, object]] = [
        {"role": "user", "content": "Where should we go dancing on Friday?"},
        {
            "role": "assistant",
            "content": "Looking for a spot.",
            "reasoning_content": _REASONING_PARAGRAPH * 60,
            "tool_calls": [
                {
                    "id": f"call_ix_{index:02d}",
                    "function": {
                        "name": "IX_verbosity",
                        "arguments": (
                            '{"evidence":"she answers in one line and asks for '
                            'the short version every time","setting":"Terse"}'
                        ),
                    },
                }
                for index in range(43)
            ],
        },
        {
            "role": "assistant",
            "content": "The Empress on Friday — it has a floor.",
            "reasoning_content": "Terse, as selected.",
        },
    ]
    return build_turn_pool(loop_messages)
