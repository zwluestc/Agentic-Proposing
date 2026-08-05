# Contributing

Thank you for your interest in Agentic Proposing. Bug reports, documentation improvements, new tool adapters, verifier extensions, and research-oriented contributions are welcome.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

## Pull requests

1. Keep changes focused and include tests for behavior changes.
2. Preserve the permission boundaries of network, process, and registry-writing tools.
3. Document new configuration fields and public commands.
4. Use synthetic or redistributable fixtures; do not commit private benchmark items, credentials, or model outputs with restricted licenses.
5. Confirm that the offline test suite passes before submitting.

Please use GitHub issues for bugs and research questions that would benefit from public discussion. Never include API keys, access tokens, private datasets, or personally identifiable information in an issue or pull request.
