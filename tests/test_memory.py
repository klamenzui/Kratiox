import pytest
from memory import MemoryDB
import uuid


@pytest.fixture
def db():
    # Verwende In-Memory-Datenbank für Tests
    return MemoryDB(":memory:")


def test_resolve_group_id_new(db):
    gid = uuid.uuid4().hex
    result = db.resolve_group_id("chat1", "user1",f"?{gid}")
    assert isinstance(result, str)
    assert result != gid
    assert len(result) == 32  # hex UUID


def test_resolve_group_id_existing(db):
    gid = uuid.uuid4().hex
    db.ensure_group(gid, "chat1", "user1", "person")
    groups = db.get_latest_facts("chat1", "user1")
    assert gid in groups
    assert groups[gid]["type"] == "person"
    result = db.resolve_group_id("chat1", "user1", f"?{gid}")
    assert result == gid
    result = db.resolve_group_id("chat1", "user1", f"{gid}")
    assert result == gid
    assert isinstance(result, str)
    assert len(result) == 32  # hex UUID


def test_store_and_get_settings(db):
    settings = {"lang": "de", "volume": 75}
    db.store_settings("user1", settings)
    loaded = db.get_settings("user1")
    assert loaded == settings

    # Überschreiben
    new_settings = {"lang": "en"}
    db.store_settings("user1", new_settings)
    loaded = db.get_settings("user1")
    assert loaded == new_settings


def test_ensure_group_creates_and_updates(db):
    gid = uuid.uuid4().hex
    db.ensure_group(gid, "chat1", "user1", "person")
    groups = db.get_latest_facts("chat1", "user1")
    assert gid in groups
    assert groups[gid]["type"] == "person"

    # Update type and set parent
    parent_id = uuid.uuid4().hex
    db.ensure_group(gid, "chat1", "user1", "event", parent_id=parent_id)
    groups = db.get_latest_facts("chat1", "user1")
    assert groups[gid]["type"] == "event"
    assert groups[gid]["parent_id"] == parent_id


def test_store_fact_and_get_latest(db):
    gid = uuid.uuid4().hex
    db.ensure_group(gid, "chat1", "user1", "person")
    db.store_fact(gid, "name", "Alice")
    db.store_fact(gid, "age", 30)

    result = db.get_latest_facts("chat1", "user1")
    assert gid in result
    assert result[gid]["name"] == "Alice"
    assert result[gid]["age"] == 30


def test_store_facts_bulk_insert(db):
    array = [
        {
            "group_id": "?a1",
            "type": "person",
            "name": "Bob"
        },
        {
            "group_id": "?a2",
            "type": "event",
            "title": "Meeting",
            "parent_id": "?a1"
        }
    ]
    db.store_facts("chat1", "user1", array)
    groups = db.get_latest_facts("chat1", "user1")

    assert len(groups) == 2
    names = [g.get("name") for g in groups.values()]
    assert "Bob" in names

    # Prüfe parent_id-Auflösung
    child = next(g for g in groups.values() if g["type"] == "event")
    assert "title" in child
    assert "parent_id" in child

def test_store_facts_bulk_update(db):
    gid = uuid.uuid4().hex
    array = [
        {
            "group_id": gid,
            "gender": "female",
            "hobby": "painting"
        },
        {
            "group_id": "?a2",
            "type": "event",
            "title": "Meeting",
            "parent_id": gid
        }
    ]
    db.ensure_group(gid, "chat1", "user1", "person")
    db.store_fact(gid, "name", "Alice")
    db.store_fact(gid, "age", 35)
    db.store_facts("chat1", "user1", array)

    result = db.get_latest_facts("chat1", "user1")

    assert len(result) == 2
    assert gid in result
    assert result[gid]["name"] == "Alice"
    assert result[gid]["age"] == 35
    assert result[gid]["gender"] == "female"
    assert result[gid]["hobby"] == "painting"
    assert result[gid]["parent_id"] is None
    child = next(g for g in result.values() if g["type"] == "event")
    assert child["type"] == "event"
    assert child["title"] == "Meeting"
    assert child["parent_id"] == gid


def test_get_latest_facts_empty(db):
    result = db.get_latest_facts("chatX", "userX")
    assert result == {}
