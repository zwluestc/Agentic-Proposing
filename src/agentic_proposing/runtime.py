from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts.model_verification import ModelVerifierGate, VerifierCommittee
from .artifacts.self_check_tool import SelfCheckArtifactTool
from .artifacts.submit_tool import SubmitArtifactTool
from .artifacts.verification import VerificationPipeline
from .config import RuntimeConfig
from .core.context import ContextConfig, ContextManager
from .core.harness import AgentHarness
from .core.hooks import HookBus, HookEvent, PermissionPolicy
from .core.store import WorkspaceLayout
from .corpus.store import EvidenceStore
from .providers.base import TeacherProvider
from .skills.extraction import PostAcceptSkillMining, SkillMiner
from .skills.registry import SkillRegistry
from .tools.base import ToolDispatcher, ToolRegistry
from .tools.core_tools import ToolSearchTool, WorkingStateEditTool
from .tools.corpus_tools import BenchmarkParseTool, CorpusIngestTool, EvidenceFetchTool, EvidenceSearchTool
from .tools.external_tools import ToolCatalogRefreshTool, ToolProviderManager, providers_from_config
from .tools.model_verifier_tools import VerifierCommitteeTool
from .tools.python_session_tools import (
    PythonSessionCloseTool,
    PythonSessionCreateTool,
    PythonSessionExecTool,
    PythonSessionManager,
)
from .tools.python_tools import PythonEvalTool, SkillPythonTool
from .tools.skill_management_tools import (
    SkillDeprecateTool,
    SkillPromoteTool,
    SkillPruneTool,
    SkillVersionTool,
)
from .tools.skill_tools import SkillComposeTool, SkillLineageTool, SkillLoadTool, SkillSearchTool, SkillStageTool
from .tools.web_backends import create_search_backend
from .tools.web_evidence_tools import EvidenceIngestUrlTool
from .tools.web_tools import HttpJsonSearchBackend, WebFetchTool, WebSearchTool
from .tools.workspace_tools import ArtifactWriteTool, WorkspaceGlobTool, WorkspaceGrepTool, WorkspaceReadTool


PermissionResolver = Callable[[dict[str, Any]], bool]


@dataclass(slots=True)
class Runtime:
    layout: WorkspaceLayout
    provider: TeacherProvider
    registry: SkillRegistry
    evidence_store: EvidenceStore
    tools: ToolRegistry
    hooks: HookBus
    dispatcher: ToolDispatcher
    context_manager: ContextManager
    harness: AgentHarness
    python_session_manager: PythonSessionManager
    tool_provider_manager: ToolProviderManager


def build_runtime(
    workspace: str | Path,
    provider: TeacherProvider,
    config: RuntimeConfig | None = None,
    *,
    permission_resolver: PermissionResolver | None = None,
    verifier_provider: TeacherProvider | None = None,
    action_observer: Callable[[dict[str, Any]], None] | None = None,
) -> Runtime:
    config = config or RuntimeConfig()
    layout = WorkspaceLayout(Path(workspace).resolve())
    layout.initialize()
    skill_registry = SkillRegistry(layout)
    evidence_store = EvidenceStore(layout.evidence / "evidence.sqlite")
    hooks = HookBus()
    tools = ToolRegistry()

    deterministic_pipeline = VerificationPipeline()
    submission_pipeline = VerificationPipeline()
    if bool(config.verification.get("require_math_spec", False)):
        from .artifacts.math_gate import MathConstructionGate
        deterministic_pipeline.gates.append(MathConstructionGate())
        submission_pipeline.gates.append(MathConstructionGate())
    verifier_committee: VerifierCommittee | None = None
    if bool(config.verification.get("model_committee_enabled", False)):
        verifier_committee = VerifierCommittee(
            verifier_provider or provider,
            voters=int(config.verification.get("model_voters", 3)),
            second_audit=bool(config.verification.get("second_audit", True)),
            trajectory_max_chars=config.verification.get("trajectory_max_chars"),
        )
        submission_pipeline.gates.append(ModelVerifierGate(verifier_committee))

    require_self_check = bool(config.verification.get("require_self_check", False))
    registered = [
        ToolSearchTool(),
        WorkingStateEditTool(),
        WorkspaceGlobTool(),
        WorkspaceReadTool(),
        WorkspaceGrepTool(),
        ArtifactWriteTool(),
        SkillSearchTool(),
        SkillLoadTool(),
        SkillComposeTool(),
        SkillStageTool(),
        SkillLineageTool(),
        BenchmarkParseTool(),
        CorpusIngestTool(),
        EvidenceSearchTool(),
        EvidenceFetchTool(),
        EvidenceIngestUrlTool(),
        PythonEvalTool(),
        SkillPythonTool(),
        PythonSessionCreateTool(),
        PythonSessionExecTool(),
        PythonSessionCloseTool(),
        WebSearchTool(),
        WebFetchTool(),
        SkillPruneTool(),
        SkillPromoteTool(),
        SkillDeprecateTool(),
        SkillVersionTool(),
        SelfCheckArtifactTool(deterministic_pipeline),
        ToolCatalogRefreshTool(),
    ]
    if verifier_committee is not None:
        registered.append(VerifierCommitteeTool(verifier_committee))
    registered.append(SubmitArtifactTool(submission_pipeline, require_self_check=require_self_check))
    for tool in registered:
        tools.register(tool)

    permission = PermissionPolicy(
        allowed=set(config.permissions.get("allowed", [])),
        denied=set(config.permissions.get("denied", [])),
        ask=set(config.permissions.get("ask", [])),
        allow_read_only=bool(config.permissions.get("allow_read_only", True)),
        allow_network=bool(config.permissions.get("allow_network", False)),
        allow_process=bool(config.permissions.get("allow_process", False)),
        allow_registry_writes=bool(config.permissions.get("allow_registry_writes", True)),
    )
    dispatcher = ToolDispatcher(
        tools,
        hooks,
        permission,
        max_parallel=int(config.tools.get("max_parallel", 8)),
        max_inline_chars=int(config.tools.get("max_inline_chars", 24_000)),
        permission_resolver=permission_resolver,
    )

    python_session_manager = PythonSessionManager(
        allowed_imports=[str(item) for item in config.python.get("allowed_imports", [])],
        default_timeout=int(config.python.get("default_timeout_seconds", 20)),
    )
    providers = providers_from_config(config.external_tools)
    tool_provider_manager = ToolProviderManager(tools, providers)

    search_backend = create_search_backend(config.web)
    if search_backend is None and config.web.get("search_endpoint"):
        search_backend = HttpJsonSearchBackend(
            str(config.web["search_endpoint"]),
            dict(config.web.get("headers", {})),
            int(config.web.get("timeout_seconds", 30)),
        )

    context_options = {
        key: value for key, value in config.context.items() if key in ContextConfig.__dataclass_fields__
    }
    context_manager = ContextManager(tools, skill_registry, hooks, ContextConfig(**context_options))
    services: dict[str, Any] = {
        "tool_registry": tools,
        "tool_provider_manager": tool_provider_manager,
        "skill_registry": skill_registry,
        "evidence_store": evidence_store,
        "python_session_manager": python_session_manager,
        "hooks": hooks,
        "source_roots": config.sources.get("roots", []),
        "web_domain_allowlist": config.web.get("allowed_domains", []),
        "web_fetch_timeout": int(config.web.get("timeout_seconds", 30)),
        "contamination_questions": config.verification.get("contamination_questions", []),
        "require_self_check": require_self_check,
    }
    if search_backend is not None:
        services["web_search_backend"] = search_backend
    if verifier_committee is not None:
        services["verifier_committee"] = verifier_committee
    if config.skill_mining.get("enabled", True):
        services["post_accept_pipeline"] = PostAcceptSkillMining(
            SkillMiner(provider, skill_registry, int(config.skill_mining.get("max_events", 300)))
        )

    def close_python_sessions(event: HookEvent, payload: dict[str, Any]) -> None:
        run_id = str(payload.get("state", {}).get("run_id", ""))
        if run_id:
            python_session_manager.close_run(run_id)
        return None

    hooks.register(HookEvent.SESSION_END, close_python_sessions)

    if bool(config.external_tools.get("refresh_on_start", False)) and providers:
        decision = permission.decide("tool_catalog_refresh", "network", False)
        if decision.allow:
            tool_provider_manager.refresh()

    harness = AgentHarness(
        provider,
        dispatcher,
        context_manager,
        layout,
        hooks,
        max_turns=int(config.harness.get("max_turns", 40)),
        temperature=float(config.harness.get("temperature", 0.2)),
        max_tokens_per_turn=int(config.harness.get("max_tokens_per_turn", 8192)),
        services=services,
        action_observer=action_observer,
    )
    return Runtime(
        layout,
        provider,
        skill_registry,
        evidence_store,
        tools,
        hooks,
        dispatcher,
        context_manager,
        harness,
        python_session_manager,
        tool_provider_manager,
    )
