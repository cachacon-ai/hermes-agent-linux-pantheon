"""Tests for terminal/file tool availability in local dev environments."""

import importlib

import pytest

from model_tools import get_tool_definitions

terminal_tool_module = importlib.import_module("tools.terminal_tool")


@pytest.fixture(autouse=True)
def _clear_caches():
    """Invalidate check_fn and tool-definitions caches before each test
    so that monkeypatched env vars / config take effect."""
    from tools.registry import invalidate_check_fn_cache
    from model_tools import _clear_tool_defs_cache
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


class TestTerminalRequirements:
    def test_local_backend_requirements(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        assert terminal_tool_module.check_terminal_requirements() is True


    def test_terminal_and_execute_code_tools_resolve_for_managed_modal(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True)
        monkeypatch.setattr(terminal_tool_module, "managed_nous_tools_enabled", lambda: True)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "modal", "modal_mode": "managed"},
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "is_managed_tool_gateway_ready",
            lambda _vendor: True,
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "execute_code" in names


class TestCheckFnTransientFailureSuppression:
    """The check_fn TTL cache should absorb transient probe failures.

    Regression coverage for #21658 / #5304: a single flaky
    ``check_terminal_requirements()`` (Docker daemon busy, probe timeout)
    must not silently strip the terminal/file toolset from a subagent. After
    a recent success, a transient False is treated as a flake; a failure with
    no recent success — or past the grace window — is honored.
    """

    @pytest.fixture(autouse=True)
    def _reset(self):
        from tools.registry import invalidate_check_fn_cache

        invalidate_check_fn_cache()
        yield
        invalidate_check_fn_cache()

    def test_transient_failure_after_success_is_suppressed(self, monkeypatch):
        import tools.registry as reg

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            # First call succeeds, second flakes (False).
            return calls["n"] == 1

        # Pin the cache clock so the TTL doesn't serve a stale entry between
        # the two probes — we want both to actually run.
        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        assert reg._check_fn_cached(flaky) is True  # records last-good
        t["now"] += reg._CHECK_FN_TTL_SECONDS + 1  # expire the TTL cache
        # Within grace window of the success → flake suppressed, stays True.
        assert reg._check_fn_cached(flaky) is True
        assert calls["n"] == 2  # the probe actually ran (not just cached)

    def test_persistent_failure_after_grace_is_honored(self, monkeypatch):
        import tools.registry as reg

        def good():
            return True

        def bad():
            return False

        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        assert reg._check_fn_cached(good) is True
        # Advance past the failure grace window, then fail.
        t["now"] += reg._CHECK_FN_FAILURE_GRACE_SECONDS + 1
        # Different fn so last-good for `good` doesn't apply; bad has no success.
        assert reg._check_fn_cached(bad) is False


    def test_grace_expiry_lets_real_outage_through(self, monkeypatch):
        import tools.registry as reg

        state = {"ok": True}

        def probe():
            return state["ok"]

        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        assert reg._check_fn_cached(probe) is True
        state["ok"] = False
        # Just past TTL, within grace → flake suppressed.
        t["now"] += reg._CHECK_FN_TTL_SECONDS + 1
        assert reg._check_fn_cached(probe) is True
        # Now move well past the grace window since the last success → honored.
        t["now"] += reg._CHECK_FN_FAILURE_GRACE_SECONDS + 1
        assert reg._check_fn_cached(probe) is False

    def test_profile_scoped_availability_does_not_cross_multiplex_profiles(
        self, tmp_path
    ):
        """Both availability caches must use the active profile as a key."""
        import tools.registry as reg
        from agent.secret_scope import (
            get_secret,
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from model_tools import _clear_tool_defs_cache, get_tool_definitions

        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        profile_a.mkdir(parents=True)
        profile_b.mkdir(parents=True)
        tool_name = "profile_scoped_availability_probe"

        def probe():
            return bool(get_secret("PROFILE_CACHE_TEST_TOKEN"))

        reg.registry.register(
            name=tool_name,
            toolset="profile-cache-test",
            schema={
                "name": tool_name,
                "description": "test-only profile-scoped availability probe",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda _args: "ok",
            check_fn=probe,
        )
        set_multiplex_active(True)
        try:
            home_a = set_hermes_home_override(str(profile_a))
            secrets_a = set_secret_scope({"PROFILE_CACHE_TEST_TOKEN": "token-a"})
            try:
                tools_a = get_tool_definitions(
                    enabled_toolsets=["profile-cache-test"],
                    quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            finally:
                reset_secret_scope(secrets_a)
                reset_hermes_home_override(home_a)

            home_b = set_hermes_home_override(str(profile_b))
            secrets_b = set_secret_scope({})
            try:
                tools_b = get_tool_definitions(
                    enabled_toolsets=["profile-cache-test"],
                    quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            finally:
                reset_secret_scope(secrets_b)
                reset_hermes_home_override(home_b)

            assert tool_name in {tool["function"]["name"] for tool in tools_a}
            assert tool_name not in {tool["function"]["name"] for tool in tools_b}
        finally:
            set_multiplex_active(False)
            reg.registry.deregister(tool_name)
            reg.invalidate_check_fn_cache()
            _clear_tool_defs_cache()

    def test_unscoped_multiplex_request_bypasses_cache(self, monkeypatch):
        """An unknown profile must never share another request's cache entry."""
        import model_tools
        import tools.registry as reg
        from agent.secret_scope import set_multiplex_active

        values = iter([True, False])
        definition_calls = {"n": 0}

        def probe():
            return next(values)

        def compute_definitions(*_args, **_kwargs):
            definition_calls["n"] += 1
            return []

        set_multiplex_active(True)
        monkeypatch.setattr(model_tools, "_compute_tool_definitions", compute_definitions)
        try:
            assert reg._check_fn_cached(probe) is True
            assert reg._check_fn_cached(probe) is False
            assert not reg._check_fn_cache

            model_tools.get_tool_definitions(quiet_mode=True)
            model_tools.get_tool_definitions(quiet_mode=True)
            assert definition_calls["n"] == 2
            assert not model_tools._tool_defs_cache
        finally:
            set_multiplex_active(False)
            reg.invalidate_check_fn_cache()
            model_tools._clear_tool_defs_cache()

    def test_profile_scoped_check_cache_is_bounded(self, monkeypatch):
        """Many multiplex profiles must not grow registry caches forever."""
        import tools.registry as reg

        scopes = iter(f"/profiles/{index}" for index in range(1_000))
        monkeypatch.setattr(reg, "check_fn_cache_scope", lambda: next(scopes))

        def available():
            return True

        for _ in range(1_000):
            assert reg._check_fn_cached(available) is True

        assert len(reg._check_fn_cache) <= reg._CHECK_FN_CACHE_MAX
        assert len(reg._check_fn_last_good) <= reg._CHECK_FN_CACHE_MAX

    def test_quiet_mode_tool_defs_recover_after_probe_failure(self, monkeypatch):
        """PAN-82: a failed first quiet-mode build must not pin missing tools
        after the availability probe recovers."""
        import model_tools
        import tools.registry as reg

        state = {"ok": False}

        def flaky_terminal_check():
            return state["ok"]

        monkeypatch.setattr(
            terminal_tool_module, "check_terminal_requirements", flaky_terminal_check
        )
        import tools as tools_pkg

        monkeypatch.setattr(
            tools_pkg, "check_file_requirements", flaky_terminal_check
        )

        t = {"now": 2000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])
        monkeypatch.setattr(model_tools.time, "monotonic", lambda: t["now"])

        reg.invalidate_check_fn_cache()
        failed = get_tool_definitions(
            enabled_toolsets=["terminal", "file"], quiet_mode=True
        )
        failed_names = {x["function"]["name"] for x in failed}
        assert "read_file" not in failed_names

        state["ok"] = True
        t["now"] += reg._CHECK_FN_TTL_SECONDS + 1
        recovered = get_tool_definitions(
            enabled_toolsets=["terminal", "file"], quiet_mode=True
        )
        recovered_names = {x["function"]["name"] for x in recovered}
        assert {"read_file", "terminal"}.issubset(recovered_names)

    def test_quiet_mode_staggered_keys_clear_on_availability_recovery(
        self, monkeypatch
    ):
        """PAN-82: a late-built outer key must not retain a False verdict after
        a sibling key's re-probe recovers the shared availability check."""
        import model_tools
        import tools.registry as reg

        tool_name = "staggered_quiet_availability_probe"
        state = {"ok": False}
        probe_calls = {"n": 0}

        def probe():
            probe_calls["n"] += 1
            return state["ok"]

        reg.registry.register(
            name=tool_name,
            toolset="staggered-cache-test",
            schema={
                "name": tool_name,
                "description": "test-only staggered quiet-mode availability probe",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda _args: "ok",
            check_fn=probe,
        )

        t = {"now": 1000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])
        monkeypatch.setattr(model_tools.time, "monotonic", lambda: t["now"])

        try:
            reg.invalidate_check_fn_cache()
            model_tools._clear_tool_defs_cache()

            # t=0: key A probes False and memoizes the empty tool list.
            first = get_tool_definitions(
                enabled_toolsets=["staggered-cache-test"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            assert tool_name not in {x["function"]["name"] for x in first}
            assert probe_calls["n"] == 1

            # t=20: key B wraps the still-cached False registry verdict under a
            # different outer key (distinct enabled_toolsets fingerprint).
            t["now"] += 20.0
            staggered = get_tool_definitions(
                enabled_toolsets=["staggered-cache-test", "terminal"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            assert tool_name not in {x["function"]["name"] for x in staggered}
            assert probe_calls["n"] == 1  # registry TTL still serving False
            assert len(model_tools._tool_defs_cache) == 2

            # t=31: probe recovers; key A expires and re-probes True. That
            # effective verdict change must clear *all* outer keys so key B
            # cannot keep serving the pre-recovery False list.
            state["ok"] = True
            t["now"] = 1000.0 + reg._CHECK_FN_TTL_SECONDS + 1.0
            recovered = get_tool_definitions(
                enabled_toolsets=["staggered-cache-test"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            assert tool_name in {x["function"]["name"] for x in recovered}
            assert probe_calls["n"] == 2

            staggered_after = get_tool_definitions(
                enabled_toolsets=["staggered-cache-test", "terminal"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            staggered_names = {x["function"]["name"] for x in staggered_after}
            assert tool_name in staggered_names
            # The recovery-triggered invalidation leaves key A uncached;
            # key B is the sole fresh entry after this second miss.
            assert len(model_tools._tool_defs_cache) == 1
        finally:
            reg.registry.deregister(tool_name)
            reg.invalidate_check_fn_cache()
            model_tools._clear_tool_defs_cache()

    def test_invalidate_check_fn_cache_clears_quiet_mode_tool_defs(
        self, monkeypatch
    ):
        """Explicit availability invalidation must drop poisoned tool-def caches."""
        import model_tools
        import tools.registry as reg

        state = {"ok": False}

        def flaky_terminal_check():
            return state["ok"]

        monkeypatch.setattr(
            terminal_tool_module, "check_terminal_requirements", flaky_terminal_check
        )
        import tools as tools_pkg

        monkeypatch.setattr(
            tools_pkg, "check_file_requirements", flaky_terminal_check
        )

        reg.invalidate_check_fn_cache()
        failed = get_tool_definitions(
            enabled_toolsets=["terminal", "file"], quiet_mode=True
        )
        assert "read_file" not in {x["function"]["name"] for x in failed}
        assert model_tools._tool_defs_cache

        state["ok"] = True
        reg.invalidate_check_fn_cache()
        assert not model_tools._tool_defs_cache

        recovered = get_tool_definitions(
            enabled_toolsets=["terminal", "file"], quiet_mode=True
        )
        recovered_names = {x["function"]["name"] for x in recovered}
        assert {"read_file", "terminal"}.issubset(recovered_names)

    def test_subagent_keeps_file_tools_through_docker_flake(self, monkeypatch):
        """End-to-end: a docker probe that flakes on the 2nd build keeps the
        file/terminal toolset available for the subagent being constructed."""
        import tools.registry as reg

        flake = {"first": True}

        def flaky_terminal_check():
            if flake["first"]:
                flake["first"] = False
                return True
            return False  # transient flake on the subagent build

        monkeypatch.setattr(
            terminal_tool_module, "check_terminal_requirements", flaky_terminal_check
        )
        # file tools delegate to the same check via tools.check_file_requirements.
        import tools as tools_pkg

        monkeypatch.setattr(
            tools_pkg, "check_file_requirements", flaky_terminal_check
        )

        t = {"now": 5000.0}
        monkeypatch.setattr(reg.time, "monotonic", lambda: t["now"])

        from model_tools import get_tool_definitions, _clear_tool_defs_cache

        reg.invalidate_check_fn_cache()
        _clear_tool_defs_cache()
        # Parent build (probe ok) → records last-good.
        parent = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        assert "read_file" in {x["function"]["name"] for x in parent}

        # Subagent build moments later: TTL expired, probe flakes False, but
        # within grace → file/terminal tools must still resolve.
        t["now"] += reg._CHECK_FN_TTL_SECONDS + 1
        _clear_tool_defs_cache()
        child = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        child_names = {x["function"]["name"] for x in child}
        assert {"read_file", "write_file", "patch", "search_files", "terminal"}.issubset(
            child_names
        )
    def test_terminal_and_execute_code_tools_resolve_for_vercel_sandbox(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "vercel_sandbox", "container_disk": 51200},
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "execute_code" in names

    def test_terminal_and_execute_code_tools_hide_for_unsupported_vercel_runtime(self, monkeypatch):
        monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "vercel_sandbox",
                "container_disk": 51200,
                "vercel_runtime": "node20",
            },
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" not in names
        assert "execute_code" not in names

    def test_terminal_and_execute_code_tools_hide_for_vercel_without_auth(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_TOKEN", raising=False)
        monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {
                "env_type": "vercel_sandbox",
                "container_disk": 51200,
                "vercel_runtime": "node22",
            },
        )
        monkeypatch.setattr(
            terminal_tool_module.importlib.util,
            "find_spec",
            lambda _name: object(),
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" not in names
        assert "execute_code" not in names
