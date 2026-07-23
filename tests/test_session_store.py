import json
import stat

import pytest

import session_store
import tools as workspace_tools


def use_temporary_store(monkeypatch, tmp_path):
    store_root = tmp_path / ".agent_sessions"
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        store_root,
    )
    return store_root


def sample_snapshot(label="default"):
    return {
        "version": 1,
        "messages": [
            {
                "type": "system",
                "data": {
                    "content": f"system-{label}",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "system",
                    "name": None,
                    "id": None,
                },
            }
        ],
        "memory_summary": f"summary-{label}",
    }


def test_save_load_round_trip_and_secure_permissions(
    tmp_path,
    monkeypatch,
):
    store_root = use_temporary_store(monkeypatch, tmp_path)
    snapshot = sample_snapshot("secure")
    synced_directories = []
    monkeypatch.setattr(
        session_store,
        "_fsync_directory",
        synced_directories.append,
    )

    session_store.save("session_01", snapshot)
    loaded_snapshot = session_store.load("session_01")

    assert loaded_snapshot == snapshot
    assert session_store.list_sessions() == ["session_01"]
    assert stat.S_IMODE(store_root.stat().st_mode) == 0o700
    assert synced_directories == [store_root]
    session_file = store_root / "session_01.json"
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600

    loaded_snapshot["memory_summary"] = "外部修改"
    assert session_store.load("session_01") == snapshot


def test_invalid_session_ids_are_rejected(
    tmp_path,
    monkeypatch,
):
    store_root = use_temporary_store(monkeypatch, tmp_path)
    invalid_ids = [
        "",
        ".",
        "..",
        "nested/session",
        r"nested\session",
        str(tmp_path / "absolute"),
        "has space",
        "a" * 65,
    ]

    for session_id in invalid_ids:
        with pytest.raises(session_store.InvalidSessionIdError):
            session_store.save(session_id, sample_snapshot())
        with pytest.raises(session_store.InvalidSessionIdError):
            session_store.load(session_id)
        with pytest.raises(session_store.InvalidSessionIdError):
            session_store.delete(session_id)

    assert not store_root.exists()


def test_failed_atomic_save_preserves_original_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    store_root = use_temporary_store(monkeypatch, tmp_path)
    original_snapshot = sample_snapshot("original")
    session_store.save("stable", original_snapshot)

    def failing_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(session_store.os, "replace", failing_replace)

    with pytest.raises(session_store.SessionStoreError, match="保存会话失败"):
        session_store.save("stable", sample_snapshot("replacement"))

    assert session_store.load("stable") == original_snapshot
    assert sorted(path.name for path in store_root.iterdir()) == [
        "stable.json"
    ]


def test_load_rejects_corrupt_oversized_and_nonregular_files(
    tmp_path,
    monkeypatch,
):
    store_root = use_temporary_store(monkeypatch, tmp_path)
    store_root.mkdir(mode=0o700)

    (store_root / "invalid_utf8.json").write_bytes(b"\xff\xfe")
    (store_root / "invalid_json.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    (store_root / "array.json").write_text(
        json.dumps(["not", "an", "object"]),
        encoding="utf-8",
    )
    (store_root / "oversized.json").write_bytes(
        b"x" * (session_store.MAX_SNAPSHOT_SIZE_BYTES + 1)
    )
    (store_root / "directory.json").mkdir()
    symlink_target = tmp_path / "outside.json"
    symlink_target.write_text("{}", encoding="utf-8")
    (store_root / "linked.json").symlink_to(symlink_target)

    for session_id in [
        "invalid_utf8",
        "invalid_json",
        "array",
        "oversized",
    ]:
        with pytest.raises(session_store.CorruptSessionError):
            session_store.load(session_id)

    for session_id in ["directory", "linked"]:
        with pytest.raises(session_store.SessionStoreError):
            session_store.load(session_id)


def test_listing_filters_entries_and_workspace_tools_hide_store(
    tmp_path,
    monkeypatch,
):
    store_root = use_temporary_store(monkeypatch, tmp_path)
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    secret_snapshot = sample_snapshot("session-secret-needle")
    session_store.save("zeta", secret_snapshot)
    session_store.save("alpha", secret_snapshot)

    (store_root / "bad.name.json").write_text("{}", encoding="utf-8")
    (store_root / ".leftover.tmp").write_text("temp", encoding="utf-8")
    (store_root / "directory.json").mkdir()
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("{}", encoding="utf-8")
    (store_root / "linked.json").symlink_to(outside_file)

    assert session_store.list_sessions() == ["alpha", "zeta"]
    assert workspace_tools.list_files.invoke({}) == "outside.json"
    with pytest.raises(ValueError):
        workspace_tools.list_files.invoke(
            {"directory": ".agent_sessions"}
        )
    with pytest.raises(ValueError):
        workspace_tools.read_file.invoke(
            {"path": ".agent_sessions/alpha.json"}
        )
    with pytest.raises(ValueError):
        workspace_tools.search_text.invoke(
            {
                "query": "session-secret-needle",
                "directory": ".agent_sessions",
            }
        )
    with pytest.raises(ValueError):
        workspace_tools.write_file.invoke(
            {
                "path": ".agent_sessions/new.json",
                "content": "{}",
            }
        )
    assert workspace_tools.search_text.invoke(
        {"query": "session-secret-needle"}
    ) == "未找到匹配结果"


def test_delete_removes_only_target_and_fsyncs_directory(
    tmp_path,
    monkeypatch,
):
    use_temporary_store(monkeypatch, tmp_path)
    session_store.save("first", sample_snapshot("first"))
    session_store.save("second", sample_snapshot("second"))
    fsync_calls = []

    def tracking_fsync(file_descriptor):
        fsync_calls.append(file_descriptor)

    monkeypatch.setattr(session_store.os, "fsync", tracking_fsync)

    session_store.delete("first")

    assert session_store.list_sessions() == ["second"]
    assert session_store.load("second") == sample_snapshot("second")
    assert len(fsync_calls) == 1
    with pytest.raises(session_store.SessionNotFoundError):
        session_store.delete("first")
