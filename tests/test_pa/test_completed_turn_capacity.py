"""Completed adapter turns may leave the prompt only at a real next-user boundary."""
import copy
from types import SimpleNamespace

import pytest

from memory.adapter import PromptHistoryEvictionPolicy
from memory.event_writer import build_turn_pool, turn_journal_for
from memory.results import WriteReceipt, WriteStatus
from memory.turn_journal import JournalRecord
from pa.agent.memory import MemoryConsolidator, PromptHistoryEvictionBlockedError
from pa.session.manager import Session, SessionManager


def make_case(tmp_path, monkeypatch, *, status=WriteStatus.committed, pending=False,
              gap=False, partial=False):
    manager = SessionManager(tmp_path)
    session = Session(key='benchmark:completed')
    session.messages = [{'role': 'user', 'content': 'request'}] + [
        {'role': 'assistant', 'content': f'large tool traffic {i} ' + 'x ' * 200}
        for i in range(17)]
    session.adapter_committed_end_index = len(session.messages) - int(partial)
    manager.save(session)
    journal = turn_journal_for(manager, session.key)
    journal.append_pending(JournalRecord.for_pool(turn_id='turn1',
        span=(1 if gap else 0, len(session.messages)), timestamp='2026-09-21T00:00:00+00:00',
        pool=build_turn_pool(session.messages)))
    if not pending:
        journal.mark_written('turn1', WriteReceipt(status=status))
    provider = SimpleNamespace()
    consolidator = MemoryConsolidator(tmp_path, provider, 'fake', provider, 'fake',
        manager, 65536, lambda **k: [], lambda: [],
        PromptHistoryEvictionPolicy.after_adapter_commit, pa_max_completion_tokens=8192,
        require_committed_adapter_writes=True)
    estimates=[]
    def estimate(s):
        estimates.append(s.prompt_history_start_index)
        return (79978 if s.prompt_history_start_index < len(s.messages) else 1000, 'offline')
    monkeypatch.setattr(consolidator, 'estimate_session_prompt_tokens', estimate)
    async def forbidden(*a, **k):
        raise AssertionError('no new memory/model calls permitted')
    monkeypatch.setattr(consolidator, 'require_file_consolidation', forbidden)
    return consolidator, session, journal, estimates


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [WriteStatus.committed, WriteStatus.accepted_no_change])
async def test_post_defers_then_next_user_evicts_exact_whole_turn(tmp_path, monkeypatch, status):
    c, s, journal, estimates = make_case(tmp_path, monkeypatch, status=status)
    messages = copy.deepcopy(s.messages)
    journal_bytes = journal.path.read_bytes()
    path = next(p for p in c.session_manager.sessions_dir.glob('*.jsonl') if '.journal.' not in p.name)
    message_rows = path.read_bytes().splitlines()[1:]
    await c.maybe_compact_prompt_history(s, stage='completed_user_turn')
    assert s.prompt_history_start_index == 0  # defer is not a claim that it fits
    assert journal.path.read_bytes() == journal_bytes
    await c.maybe_compact_prompt_history(s, stage='incoming_user_turn')
    assert s.prompt_history_start_index == len(messages) == 18
    assert estimates[-1] == 18  # must re-estimate after eviction
    assert s.adapter_committed_end_index == 18
    assert s.messages == messages
    assert path.read_bytes().splitlines()[1:] == message_rows
    assert journal.path.read_bytes() == journal_bytes
    assert s.get_history(max_messages=0) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['pending', 'gap', 'partial', 'queued', 'no_journal', 'nonuser'])
@pytest.mark.parametrize('stage', ['completed_user_turn', 'incoming_user_turn'])
async def test_no_eviction_or_deferral_without_full_commit_proof(tmp_path, monkeypatch, damage, stage):
    c, s, journal, _ = make_case(tmp_path, monkeypatch,
        pending=damage=='pending', gap=damage=='gap', partial=damage=='partial',
        status=WriteStatus.queued if damage=='queued' else WriteStatus.committed)
    if damage=='no_journal':journal.path.unlink()
    if damage=='nonuser':s.messages[0]['role']='assistant'
    messages=copy.deepcopy(s.messages)
    with pytest.raises(PromptHistoryEvictionBlockedError):
        await c.maybe_compact_prompt_history(s, stage=stage)
    assert s.prompt_history_start_index == 0 and s.messages == messages


@pytest.mark.asyncio
async def test_default_system_forced_replay_path_still_refuses_capacity(tmp_path, monkeypatch):
    c, s, _, _ = make_case(tmp_path, monkeypatch)
    with pytest.raises(PromptHistoryEvictionBlockedError, match='CAPACITY'):
        await c.maybe_compact_prompt_history(s)
    assert s.prompt_history_start_index == 0


@pytest.mark.asyncio
async def test_existing_below_capacity_behavior_does_not_change(tmp_path, monkeypatch):
    c, s, _, _ = make_case(tmp_path, monkeypatch)
    monkeypatch.setattr(c, 'estimate_session_prompt_tokens', lambda _: (25000, 'offline'))
    await c.maybe_compact_prompt_history(s, stage='incoming_user_turn')
    assert s.prompt_history_start_index == 0


@pytest.mark.asyncio
async def test_no_fake_fits_if_base_prompt_remains_oversized(tmp_path, monkeypatch):
    c, s, journal, _ = make_case(tmp_path, monkeypatch)
    monkeypatch.setattr(c, 'estimate_session_prompt_tokens', lambda _: (79978, 'offline'))
    with pytest.raises(PromptHistoryEvictionBlockedError, match='CAPACITY'):
        await c.maybe_compact_prompt_history(s, stage='incoming_user_turn')
    assert s.prompt_history_start_index == 18
    assert len(s.messages) == 18  # only the prompt cursor changed


@pytest.mark.asyncio
async def test_invalid_stage_rejected_before_any_changes(tmp_path, monkeypatch):
    c, s, _, _ = make_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='lifecycle'):
        await c.maybe_compact_prompt_history(s, stage='both')
    assert s.prompt_history_start_index == 0


@pytest.mark.asyncio
async def test_probe_skip_and_loop_stage_passthrough(tmp_path, monkeypatch):
    from pa.agent.loop import AgentLoop
    c, s, _, _ = make_case(tmp_path, monkeypatch)
    loop = object.__new__(AgentLoop)
    loop.memory_consolidator=c
    loop.skip_consolidation=True
    await loop._compact_prompt_history_if_needed(s, stage='incoming_user_turn')
    await loop._compact_prompt_history_after_turn(s, stage='completed_user_turn')
    assert s.prompt_history_start_index == 0
    loop.skip_consolidation=False
    await loop._compact_prompt_history_after_turn(s, stage='completed_user_turn')
    assert s.prompt_history_start_index == 0
    await loop._compact_prompt_history_if_needed(s, stage='incoming_user_turn')
    assert s.prompt_history_start_index == 18
