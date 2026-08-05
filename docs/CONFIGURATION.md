# Runtime configuration

The harness is intentionally provider-agnostic. A research workspace owns one `config.json`; credentials should normally be supplied through environment variables rather than committed into the file.

## Permissions

```json
{
  "permissions": {
    "allow_read_only": true,
    "allow_network": false,
    "allow_process": false,
    "allow_registry_writes": true,
    "allowed": [],
    "denied": [],
    "ask": ["evidence_ingest_url"]
  }
}
```

Precedence is `denied -> ask -> explicitly allowed -> side-effect policy -> read-only policy`. An `ask` entry is approved only when the embedding application passes a `permission_resolver` to `build_runtime`; batch CLI runs reject unanswered approvals safely.

## Web search and evidence acquisition

`web_search` returns discovery pointers. Those pointers become citable local evidence only after `evidence_ingest_url` fetches, hashes, timestamps, chunks, and indexes a page. Public-page fetching blocks private, loopback, link-local, and reserved addresses and revalidates redirect targets.

Brave:

```json
{
  "web": {
    "backend": "brave",
    "api_key_env": "BRAVE_SEARCH_API_KEY",
    "allowed_domains": ["who.int", "nih.gov"],
    "timeout_seconds": 30
  }
}
```

Tavily uses `"backend": "tavily"` and defaults to `TAVILY_API_KEY`. A self-hosted SearXNG instance uses:

```json
{
  "web": {
    "backend": "searxng",
    "endpoint": "https://search.example.org",
    "allowed_domains": ["who.int", "nih.gov"]
  }
}
```

Network tools additionally require `permissions.allow_network=true` or a narrow explicit allow.

## Persistent Python sessions

```json
{
  "python": {
    "allowed_imports": ["math", "statistics", "numpy", "pandas"],
    "default_timeout_seconds": 20
  },
  "permissions": {"allow_process": true}
}
```

The sequence is `python_session_create -> python_session_exec* -> python_session_close`. Variables persist within one run and all remaining workers are closed at session end. AST restrictions, an isolated interpreter flag, a stripped environment, per-run working directories, timeouts, and import allowlists reduce accidental risk. This is not an OS/container boundary for hostile code.

## Dynamic external tools

Static JSON-over-HTTP definitions and remotely served catalogs use the same dispatcher, permission, hook, event, deferred-discovery, and output-spill path as built-in tools.

```json
{
  "external_tools": {
    "refresh_on_start": false,
    "providers": [
      {
        "id": "medical-services",
        "catalog_url": "https://tools.example.org/catalog",
        "headers": {"Authorization": "$ENV:MED_TOOL_TOKEN"},
        "timeout_seconds": 30,
        "tools": [
          {
            "name": "terminology_lookup",
            "description": "Resolve a controlled medical term.",
            "endpoint": "https://tools.example.org/terminology",
            "method": "POST",
            "input_schema": {
              "type": "object",
              "properties": {"term": {"type": "string"}},
              "required": ["term"]
            },
            "side_effect": "network",
            "read_only": true,
            "deferred": true
          }
        ]
      }
    ]
  }
}
```

`tool_catalog_refresh` validates all providers and names before changing the registry, so refresh is transactional. Built-ins cannot be overridden and external names must be globally unique. This is a general provider boundary suitable for adding an MCP transport adapter; the current repository implements JSON-over-HTTP catalogs and calls, not the full MCP protocol lifecycle.

## Verification and context

```json
{
  "tools": {"max_parallel": 8, "max_inline_chars": 24000},
  "context": {
    "max_context_chars": 100000,
    "max_transcript_chars": 45000,
    "max_skill_chars_each": 12000,
    "max_skill_total_chars": 30000
  },
  "verification": {
    "require_self_check": true,
    "model_committee_enabled": false,
    "model_voters": 3,
    "second_audit": true
  }
}
```

With freshness enforcement enabled, `self_check_artifact` records a fingerprint of the exact candidate. Any material edit invalidates it. `submit_artifact` reruns the full submission pipeline and is the only terminal action; an optional model committee is intentionally reserved for this final gate rather than every intermediate check.
