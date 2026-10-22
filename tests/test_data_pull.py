"""`mempabench data pull` copies every listed persona data file into the
checkout, overwriting what is there, or stops naming the listed files the
download lacks. No network: `snapshot_download` returns a fake Hugging Face
cache snapshot whose files are symlinks into the cache repo's `blobs/`."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.utils import reset_sessions

from mempabench import cli

CACHE_REPO_NAME = "datasets--" + cli.DATASET_REPO.replace("/", "--")
REAL_SNAPSHOT_DOWNLOAD = huggingface_hub.snapshot_download


def _persona_files(persona: str) -> dict[str, bytes]:
    return {
        f"data/personas/{persona}_identity.yaml": f"name: {persona}\n".encode(),
        f"data/state_fixtures/{persona}/world.json": b"{}\n",
        f"Authoring/probe/drafts/{persona}/fin_01/fixture/.gitkeep": b"",
    }


def _tree(root: Path) -> list[str]:
    """Every path under `root` (files, directories, symlinks), relative and sorted."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


@pytest.fixture
def fake_pull(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fake cache snapshot at `DATASET_REVISION`, its file list, and an empty checkout."""
    cache_repo = tmp_path / "hub" / CACHE_REPO_NAME
    snapshot = cache_repo / "snapshots" / cli.DATASET_REVISION
    (cache_repo / "blobs").mkdir(parents=True)
    lines = [f"revision\t{cli.DATASET_REVISION}"]
    for persona in cli.PERSONAS:
        for relative, data in sorted(_persona_files(persona).items()):
            blob = cache_repo / "blobs" / f"blob{len(lines)}"
            blob.write_bytes(data)
            link = snapshot / persona / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(blob, link.parent))
            lines.append(f"{persona}\t{relative}")
    file_list = tmp_path / "persona_data_files.tsv"
    file_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    root = tmp_path / "checkout"
    root.mkdir()
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(cli, "PERSONA_DATA_FILES", file_list)
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    return SimpleNamespace(tmp=tmp_path, snapshot=snapshot, file_list=file_list, root=root, calls=calls)


def test_pull_copies_every_listed_file_from_cache_symlinks(fake_pull, capsys):
    assert cli.pull(list(cli.PERSONAS)) == 0
    assert fake_pull.calls == [
        {
            "repo_id": cli.DATASET_REPO,
            "repo_type": "dataset",
            "revision": cli.DATASET_REVISION,
            "allow_patterns": [f"{persona}/**" for persona in cli.PERSONAS],
        }
    ]
    expected = {relative: data for persona in cli.PERSONAS for relative, data in _persona_files(persona).items()}
    copied = {path for path in _tree(fake_pull.root) if (fake_pull.root / path).is_file()}
    assert copied == set(expected)
    for relative, data in expected.items():
        target = fake_pull.root / relative
        assert not target.is_symlink() and target.read_bytes() == data
    assert "12 files copied" in capsys.readouterr().out


def test_pull_overwrites_an_existing_file_with_the_pinned_bytes(fake_pull):
    target = fake_pull.root / "data/personas/user_a_identity.yaml"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"name: edited locally\n")
    assert cli.pull(list(cli.PERSONAS)) == 0
    assert target.read_bytes() == b"name: user_a\n"


def test_a_persona_pull_downloads_and_copies_only_that_persona(fake_pull, capsys):
    for persona in ("user_b", "user_c", "user_d"):  # not downloaded, so not in the snapshot
        shutil.rmtree(fake_pull.snapshot / persona)
    assert cli.main(["data", "pull", "--persona", "user_a"]) == 0
    assert [call["allow_patterns"] for call in fake_pull.calls] == [["user_a/**"]]
    copied = {path for path in _tree(fake_pull.root) if (fake_pull.root / path).is_file()}
    assert copied == set(_persona_files("user_a"))
    assert "3 files copied" in capsys.readouterr().out


def test_a_persona_pull_missing_one_of_that_personas_files_names_it_and_writes_nothing(fake_pull):
    for persona in ("user_b", "user_c", "user_d"):  # not downloaded, so not in the snapshot
        shutil.rmtree(fake_pull.snapshot / persona)
    (fake_pull.snapshot / "user_a/data/state_fixtures/user_a/world.json").unlink()
    with pytest.raises(
        cli.PersonaDataPullError,
        match=r"nothing was written under .*; the snapshot .* lacks 1 listed file\(s\):\n"
        r"user_a/data/state_fixtures/user_a/world\.json$",
    ):
        cli.main(["data", "pull", "--persona", "user_a"])
    assert _tree(fake_pull.root) == []


@pytest.fixture
def offline_cache(fake_pull, monkeypatch):
    """The real `snapshot_download` with HF_HUB_OFFLINE=1, reading the fake cache.

    `huggingface_hub` reads HF_HUB_OFFLINE into `constants` at import, and a
    session made while it is set refuses every request; no request is sent.
    """
    monkeypatch.setattr(huggingface_hub, "snapshot_download", REAL_SNAPSHOT_DOWNLOAD)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(fake_pull.tmp / "hub"))
    reset_sessions()
    yield
    reset_sessions()


def test_an_offline_pull_from_a_complete_cache_copies_every_listed_file(fake_pull, offline_cache, capsys):
    assert cli.main(["data", "pull"]) == 0
    expected = {relative: data for persona in cli.PERSONAS for relative, data in _persona_files(persona).items()}
    copied = {path for path in _tree(fake_pull.root) if (fake_pull.root / path).is_file()}
    assert copied == set(expected)
    assert all((fake_pull.root / relative).read_bytes() == data for relative, data in expected.items())
    assert "12 files copied" in capsys.readouterr().out


@pytest.mark.parametrize("lost", ["blob", "snapshot file"])
def test_an_offline_pull_from_an_incomplete_cache_names_the_missing_file_and_writes_nothing(
    fake_pull, offline_cache, lost
):
    snapshot_file = fake_pull.snapshot / "user_b/data/state_fixtures/user_b/world.json"
    (snapshot_file.resolve() if lost == "blob" else snapshot_file).unlink()
    with pytest.raises(cli.PersonaDataPullError, match=r"\nuser_b/data/state_fixtures/user_b/world\.json$"):
        cli.pull(list(cli.PERSONAS))
    assert _tree(fake_pull.root) == []


def test_pull_refuses_a_snapshot_missing_a_listed_file_and_writes_nothing(fake_pull):
    (fake_pull.snapshot / "user_b/data/state_fixtures/user_b/world.json").unlink()
    with pytest.raises(cli.PersonaDataPullError, match=r"\nuser_b/data/state_fixtures/user_b/world\.json$"):
        cli.pull(list(cli.PERSONAS))
    assert _tree(fake_pull.root) == []


def _is_persona_data_path(persona: str, relative: str) -> bool:
    """Whether `relative` (a path under the checkout root) is one a persona's data occupies."""
    files = {
        f"data/personas/{persona}_identity.yaml",
        f"data/personas/{persona}_preferences.yaml",
    }
    folders = (
        f"data/state_fixtures/{persona}/",
        f"simulator/profiles/{persona}/",
        f"Authoring/outline/drafts/{persona}/",
        f"Authoring/sessions/normal/drafts/{persona}/",
        f"Authoring/sessions/forced/drafts/{persona}/",
        f"Authoring/sessions/forced/transcripts/{persona}/",
        f"Authoring/probe/outline/{persona}/",
        f"Authoring/probe/drafts/{persona}/",
    )
    return relative in files or relative.startswith(folders)


def test_the_committed_file_list_is_well_formed_for_the_pinned_dataset_revision():
    """A DATASET_REVISION bump without a regenerated file list, or a list line
    that `data pull` would copy outside a persona's data, fails here."""
    lines = cli.PERSONA_DATA_FILES.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"revision\t{cli.DATASET_REVISION}"
    seen: dict[str, set[str]] = {persona: set() for persona in cli.PERSONAS}
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        assert len(fields) == 2, f"line {number}: {line!r}"
        persona, relative = fields
        assert persona in seen, f"line {number}: unknown persona {persona!r}"
        assert all(part not in ("", ".", "..") for part in relative.split("/")), f"line {number}: {relative!r}"
        assert _is_persona_data_path(persona, relative), f"line {number}: not {persona}'s data: {relative!r}"
        assert relative not in seen[persona], f"line {number}: {relative} is listed twice"
        seen[persona].add(relative)
    assert all(seen.values())
    revision, files = cli.load_persona_data_files(cli.PERSONA_DATA_FILES)
    assert revision == cli.DATASET_REVISION
    assert {persona: set(paths) for persona, paths in files.items()} == seen
