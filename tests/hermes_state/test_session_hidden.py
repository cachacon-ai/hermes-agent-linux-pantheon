import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def test_hidden_excluded_by_default_included_on_request(db):
    db.create_session("visible", source="cli")
    db.create_session("secret", source="cli")
    # Give both a message so the default min_message_count filter keeps them.
    for sid in ("visible", "secret"):
        db._conn.execute(
            "UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,)
        )
    db._conn.commit()

    # Flip the hidden flag on one session.
    assert db.set_session_hidden("secret", True) is True
    assert db.get_session("secret")["hidden"] == 1
    assert db.get_session("visible")["hidden"] == 0

    # Default listing drops the hidden row; include_hidden=True surfaces it.
    default_ids = {s["id"] for s in db.list_sessions_rich(min_message_count=1)}
    assert default_ids == {"visible"}

    all_ids = {
        s["id"]
        for s in db.list_sessions_rich(min_message_count=1, include_hidden=True)
    }
    assert all_ids == {"visible", "secret"}

    # Unhiding brings it back into the default listing.
    assert db.set_session_hidden("secret", False) is True
    assert db.get_session("visible")["hidden"] == 0
    unhidden_ids = {s["id"] for s in db.list_sessions_rich(min_message_count=1)}
    assert unhidden_ids == {"visible", "secret"}


def _paged_ids(db, **kwargs):
    """Walk list_sessions_rich pages to exhaustion (tiny pages exercise the
    page boundaries that page-against-total callers rely on). ``kwargs`` must
    be filters both list_sessions_rich and session_count accept."""
    ids = []
    offset = 0
    while True:
        page = db.list_sessions_rich(limit=1, offset=offset, **kwargs)
        ids.extend(s["id"] for s in page)
        if len(page) < 1 or offset + len(page) >= db.session_count(**kwargs):
            break
        offset += 1
    return ids


def test_session_count_matches_paged_listing_with_hidden_and_archived(db):
    # Mix of visible/hidden and archived rows: the count must always agree
    # with the rows a page-against-total caller can actually reach, for both
    # inclusion modes. Guards the PAN-2 class: count included hidden rows the
    # listing filtered out, so callers under-fetched pages and lost rows.
    db.create_session("visible", source="cli")
    db.create_session("hidden-live", source="cli")
    db.create_session("hidden-archived", source="cli")
    db.create_session("visible-archived", source="cli")
    for sid in ("visible", "hidden-live", "hidden-archived", "visible-archived"):
        db._conn.execute("UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,))
    db._conn.commit()
    db.set_session_hidden("hidden-live", True)
    db.set_session_hidden("hidden-archived", True)
    db.set_session_archived("hidden-archived", True)
    db.set_session_archived("visible-archived", True)

    filters = dict(min_message_count=1, include_archived=True)
    paged_default = _paged_ids(db, **filters)
    assert db.session_count(**filters) == len(paged_default)
    assert set(paged_default) == {"visible", "visible-archived"}

    filters_hidden = dict(filters, include_hidden=True)
    paged_all = _paged_ids(db, **filters_hidden)
    assert db.session_count(**filters_hidden) == len(paged_all)
    assert set(paged_all) == {
        "visible", "visible-archived", "hidden-live", "hidden-archived",
    }

    # Detail reads stay hidden-agnostic regardless of listing mode.
    assert db.get_session("hidden-live")["hidden"] == 1
