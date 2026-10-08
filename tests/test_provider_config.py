import json

from agentic_proposing.config import create_provider, resolve_provider_config


def test_local_credentials_and_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / "local.json"
    path.write_text(json.dumps({"api_key": "local-secret", "base_url": "https://local.example/v1"}))
    config = {"credential_file": str(path), "api_key_env": "TEST_PROVIDER_KEY",
              "base_url_env": "TEST_PROVIDER_URL", "model": "example"}
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    monkeypatch.delenv("TEST_PROVIDER_URL", raising=False)
    resolved = resolve_provider_config(config)
    assert resolved["api_key"] == "local-secret"
    assert resolved["base_url"] == "https://local.example/v1"
    assert "api_key" not in config
    monkeypatch.setenv("TEST_PROVIDER_KEY", "env-secret")
    monkeypatch.setenv("TEST_PROVIDER_URL", "https://env.example/v1")
    provider = create_provider(config)
    assert provider.api_key == "env-secret"
    assert provider.base_url == "https://env.example/v1"
    assert resolve_provider_config(dict(config, api_key="explicit-secret"))["api_key"] == "explicit-secret"


def test_missing_local_file_allows_portable_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "portable-secret")
    resolved = resolve_provider_config({"credential_file": str(tmp_path / "absent.json"),
                                        "api_key_env": "TEST_PROVIDER_KEY"})
    assert resolved["api_key"] == "portable-secret"
