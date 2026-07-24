"""Exact-surface adapter for the generic retrospective contamination audit.

The generic contamination package audits one public artifact against a private comparison
namespace.  Agentic Replay treats the canonical model-visible workspace surface as that single
public artifact.  This adapter fixes the naming and binding conventions and rejects incomplete or
extra audit material.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.admission import (
    AgenticAdmissionError,
    AgenticContaminationBinding,
    VerifiedContaminationAudit,
)
from vaxreplay.agentic.schema import AgenticTaskEnvelope
from vaxreplay.agentic.workspace import (
    LoadedAgenticWorkspace,
    load_agentic_workspace,
    parse_model_visible_surface_bytes,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.contamination import (
    AuditDisposition,
    ContaminationAuditManifest,
    ContaminationAuditPolicy,
    IdentifierNeedle,
    audit_manifest_sha256,
    make_audit_input,
    model_sha256,
    verify_contamination_audit,
)

AGENTIC_AUDIT_PUBLIC_ARTIFACT_ID = 'agentic-model-visible-surface'
AGENTIC_AUDIT_MANIFEST_KEY = 'audit-manifest.json'
AGENTIC_AUDIT_POLICY_KEY = 'audit-policy.json'
AGENTIC_AUDIT_IDENTIFIERS_KEY = 'identifiers.json'
AGENTIC_AUDIT_PROTECTED_CORPUS_KEY = 'protected-corpus-manifest.json'
AGENTIC_AUDIT_COMPARISON_PREFIX = 'comparisons/'
AGENTIC_AUDIT_VERIFIER_ID = 'vaxreplay.agentic.exact-surface-contamination'
AGENTIC_AUDIT_VERIFIER_VERSION = '0.1'

_SAFE_ARTIFACT_ID = re.compile(r'^[a-z0-9][a-z0-9._-]{0,127}$')


class AgenticProtectedCorpusArtifact(StrictModel):
    artifact_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    byte_count: int = Field(gt=0)
    category: Literal['post_cutoff_source', 'outcome_or_label', 'protected_identifier']
    source_uri: str = Field(min_length=1)
    acquired_at: datetime

    @field_validator('artifact_id')
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        _require_safe_artifact_id(value)
        return value

    @field_validator('acquired_at')
    @classmethod
    def validate_acquired_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('protected artifact acquired_at must include a UTC offset')
        return value.astimezone(UTC)


class AgenticProtectedCorpusManifest(StrictModel):
    """Exact organizer-selected comparison corpus, not a claim of global completeness."""

    schema_version: Literal['vaxreplay.agentic-protected-corpus-manifest.v0.1'] = (
        'vaxreplay.agentic-protected-corpus-manifest.v0.1'
    )
    corpus_id: str = Field(min_length=1)
    historical_cutoff: datetime
    selection_policy_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    scope_description: str = Field(min_length=1, max_length=10_000)
    coverage_limitations: str = Field(min_length=1, max_length=10_000)
    organizer_inventory_complete_under_policy: Literal[True] = True
    proves_global_completeness: Literal[False] = False
    artifacts: tuple[AgenticProtectedCorpusArtifact, ...] = Field(min_length=1)

    @field_validator('historical_cutoff')
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('protected corpus historical_cutoff must include a UTC offset')
        return value.astimezone(UTC)

    @model_validator(mode='after')
    def validate_artifacts(self) -> Self:
        artifact_ids = tuple(artifact.artifact_id for artifact in self.artifacts)
        if artifact_ids != tuple(sorted(artifact_ids)) or len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError('protected corpus artifacts must use unique IDs in sorted order')
        return self


def agentic_case_universe_sha256(
    *,
    workspace_manifest_sha256: str,
    model_visible_surface_sha256: str,
) -> str:
    """Bind the one-workspace audit universe without exposing organizer-private source metadata."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                'schema_version': 'vaxreplay.agentic-contamination-universe.v0.1',
                'workspace_manifest_sha256': workspace_manifest_sha256,
                'model_visible_surface_sha256': model_visible_surface_sha256,
            }
        )
    ).hexdigest()


def make_agentic_audit_input(
    workspace: LoadedAgenticWorkspace,
    *,
    protected_corpus: AgenticProtectedCorpusManifest,
    comparison_payloads: Mapping[str, bytes],
):
    """Create the generic audit input for one exact, currently loaded workspace surface."""

    workspace = load_agentic_workspace(workspace.root)
    _require_protected_corpus(workspace, protected_corpus, comparison_payloads)
    return make_audit_input(
        case_id=workspace.manifest_sha256,
        episode_id=workspace.task.episode_id,
        decision_package_sha256=workspace.manifest.workspace_tree_sha256,
        episode_manifest_sha256=workspace.task.episode_manifest_sha256,
        public_artifact_id=AGENTIC_AUDIT_PUBLIC_ARTIFACT_ID,
        public_payload=workspace.model_visible_surface,
        comparison_payloads=comparison_payloads,
    )


def make_agentic_contamination_binding(
    workspace: LoadedAgenticWorkspace,
    *,
    manifest: ContaminationAuditManifest,
    policy: ContaminationAuditPolicy,
    protected_corpus: AgenticProtectedCorpusManifest,
) -> AgenticContaminationBinding:
    """Bind a pass manifest to one exact workspace; verification remains a separate step."""

    workspace = load_agentic_workspace(workspace.root)
    if protected_corpus.selection_policy_sha256 != workspace.build_policy.protected_outcome_namespace_sha256:
        raise AgenticAdmissionError('protected corpus does not use the outcome namespace committed by the build policy')
    if len(manifest.audits) != 1:
        raise AgenticAdmissionError('an Agentic workspace contamination manifest must contain exactly one audit')
    audit = manifest.audits[0]
    _require_agentic_audit_identity(workspace, audit.audit_input)
    _require_protected_bindings(protected_corpus, audit.audit_input.comparison_artifacts)
    expected_universe = agentic_case_universe_sha256(
        workspace_manifest_sha256=workspace.manifest_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
    )
    if manifest.case_universe_sha256 != expected_universe:
        raise AgenticAdmissionError('contamination manifest is bound to a different Agentic workspace universe')
    if audit.disposition != AuditDisposition.PASS:
        raise AgenticAdmissionError('only a pass audit can create a workspace contamination binding')
    if manifest.policy_sha256 != model_sha256(policy):
        raise AgenticAdmissionError('contamination manifest does not use the supplied pinned audit policy')
    return AgenticContaminationBinding(
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        contamination_audit_manifest_sha256=audit_manifest_sha256(manifest),
        contamination_audit_policy_sha256=model_sha256(policy),
        protected_corpus_manifest_sha256=hashlib.sha256(canonical_json_bytes(protected_corpus)).hexdigest(),
        protected_outcome_namespace_sha256=protected_corpus.selection_policy_sha256,
        audited_file_count=len(workspace.manifest.entries),
    )


def verify_agentic_contamination_audit(
    binding: AgenticContaminationBinding,
    *,
    model_visible_surface: bytes,
    audit_artifacts: Mapping[str, bytes],
) -> VerifiedContaminationAudit:
    """Rebuild a fixed audit and prove that it covered every visible path and byte.

    ``audit_artifacts`` must contain canonical ``audit-manifest.json``, ``audit-policy.json``, and
    ``identifiers.json`` plus exactly one ``comparisons/<artifact-id>`` entry for each comparison
    artifact committed by the audit input.  Any missing or extra key fails closed.
    """

    surface_sha256 = hashlib.sha256(model_visible_surface).hexdigest()
    if surface_sha256 != binding.model_visible_surface_sha256:
        raise AgenticAdmissionError('contamination verifier received a different model-visible surface')
    files, task = _parse_model_visible_surface(model_visible_surface)
    if len(files) != binding.audited_file_count:
        raise AgenticAdmissionError('contamination audit file count does not match the canonical surface')

    manifest_bytes = _required_artifact(audit_artifacts, AGENTIC_AUDIT_MANIFEST_KEY)
    policy_bytes = _required_artifact(audit_artifacts, AGENTIC_AUDIT_POLICY_KEY)
    identifiers_bytes = _required_artifact(audit_artifacts, AGENTIC_AUDIT_IDENTIFIERS_KEY)
    protected_corpus_bytes = _required_artifact(audit_artifacts, AGENTIC_AUDIT_PROTECTED_CORPUS_KEY)
    try:
        manifest = ContaminationAuditManifest.model_validate_json(manifest_bytes)
        policy = ContaminationAuditPolicy.model_validate_json(policy_bytes)
        raw_identifiers = json.loads(identifiers_bytes)
        if not isinstance(raw_identifiers, list):
            raise ValueError('identifiers must be a JSON list')
        identifiers = tuple(
            IdentifierNeedle.model_validate_json(canonical_json_bytes(value)) for value in raw_identifiers
        )
        protected_corpus = AgenticProtectedCorpusManifest.model_validate_json(protected_corpus_bytes)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise AgenticAdmissionError(f'invalid contamination audit artifacts: {error}') from error
    if manifest_bytes != canonical_json_bytes(manifest):
        raise AgenticAdmissionError('contamination manifest must use canonical JSON')
    if policy_bytes != canonical_json_bytes(policy):
        raise AgenticAdmissionError('contamination policy must use canonical JSON')
    if identifiers_bytes != canonical_json_bytes([value.model_dump(mode='json') for value in identifiers]):
        raise AgenticAdmissionError('contamination identifiers must use canonical JSON')
    if protected_corpus_bytes != canonical_json_bytes(protected_corpus):
        raise AgenticAdmissionError('protected corpus manifest must use canonical JSON')
    if audit_manifest_sha256(manifest) != binding.contamination_audit_manifest_sha256:
        raise AgenticAdmissionError('contamination manifest does not match the workspace binding')
    if len(manifest.audits) != 1:
        raise AgenticAdmissionError('an Agentic contamination manifest must contain exactly one audit')
    if manifest.policy_sha256 != model_sha256(policy):
        raise AgenticAdmissionError('contamination policy does not match the manifest commitment')
    if model_sha256(policy) != binding.contamination_audit_policy_sha256:
        raise AgenticAdmissionError('contamination audit policy is not the release-bound policy')
    if hashlib.sha256(protected_corpus_bytes).hexdigest() != binding.protected_corpus_manifest_sha256:
        raise AgenticAdmissionError('protected corpus manifest is not the release-bound corpus')
    if protected_corpus.selection_policy_sha256 != binding.protected_outcome_namespace_sha256:
        raise AgenticAdmissionError('protected corpus does not match the bound outcome namespace policy')

    audit = manifest.audits[0]
    expected_universe = agentic_case_universe_sha256(
        workspace_manifest_sha256=binding.workspace_manifest_sha256,
        model_visible_surface_sha256=binding.model_visible_surface_sha256,
    )
    if manifest.case_universe_sha256 != expected_universe:
        raise AgenticAdmissionError('contamination manifest is bound to a different case universe')
    if (
        audit.audit_input.case_id != binding.workspace_manifest_sha256
        or audit.audit_input.episode_id != task.episode_id
        or audit.audit_input.decision_package_sha256 != binding.workspace_tree_sha256
        or audit.audit_input.episode_manifest_sha256 != task.episode_manifest_sha256
        or audit.audit_input.public_artifact.artifact_id != AGENTIC_AUDIT_PUBLIC_ARTIFACT_ID
        or audit.audit_input.public_artifact.sha256 != binding.model_visible_surface_sha256
        or audit.audit_input.public_artifact.byte_count != len(model_visible_surface)
    ):
        raise AgenticAdmissionError('contamination audit input is not bound to the exact Agentic workspace')

    comparison_payloads: dict[str, bytes] = {}
    for comparison in audit.audit_input.comparison_artifacts:
        _require_safe_artifact_id(comparison.artifact_id)
        key = AGENTIC_AUDIT_COMPARISON_PREFIX + comparison.artifact_id
        comparison_payloads[comparison.artifact_id] = _required_artifact(audit_artifacts, key)
    _require_protected_corpus_for_task(task, protected_corpus, comparison_payloads)
    _require_protected_bindings(protected_corpus, audit.audit_input.comparison_artifacts)
    expected_keys = {
        AGENTIC_AUDIT_MANIFEST_KEY,
        AGENTIC_AUDIT_POLICY_KEY,
        AGENTIC_AUDIT_IDENTIFIERS_KEY,
        AGENTIC_AUDIT_PROTECTED_CORPUS_KEY,
        *(AGENTIC_AUDIT_COMPARISON_PREFIX + artifact_id for artifact_id in comparison_payloads),
    }
    if set(audit_artifacts) != expected_keys:
        missing = sorted(expected_keys - set(audit_artifacts))
        extra = sorted(set(audit_artifacts) - expected_keys)
        raise AgenticAdmissionError(
            f'contamination audit artifact inventory mismatch; missing={missing}, extra={extra}'
        )
    try:
        verify_contamination_audit(
            audit,
            policy=policy,
            public_payload=model_visible_surface,
            comparison_payloads=comparison_payloads,
            identifiers=identifiers,
        )
    except ValueError as error:
        raise AgenticAdmissionError(f'contamination audit verification failed: {error}') from error
    if audit.disposition != AuditDisposition.PASS:
        raise AgenticAdmissionError('Agentic workspace contamination audit did not pass')

    return VerifiedContaminationAudit(
        contamination_audit_manifest_sha256=binding.contamination_audit_manifest_sha256,
        contamination_audit_policy_sha256=binding.contamination_audit_policy_sha256,
        protected_corpus_manifest_sha256=binding.protected_corpus_manifest_sha256,
        audited_surface_sha256=surface_sha256,
        audited_file_count=len(files),
        judge_count=len(audit.judge_runs),
        verifier_id=AGENTIC_AUDIT_VERIFIER_ID,
        verifier_version=AGENTIC_AUDIT_VERIFIER_VERSION,
    )


def _parse_model_visible_surface(payload: bytes) -> tuple[tuple[dict[str, str], ...], AgenticTaskEnvelope]:
    try:
        parsed = parse_model_visible_surface_bytes(payload)
    except ValueError as error:
        raise AgenticAdmissionError(f'model-visible surface framing is invalid: {error}') from error
    files = [{'path': path, 'utf8_content': content.decode('utf-8')} for path, content in parsed.items()]
    task_records = [item for item in files if item['path'] == 'TASK.json']
    if len(task_records) != 1:
        raise AgenticAdmissionError('model-visible surface must contain exactly one TASK.json')
    try:
        task_bytes = task_records[0]['utf8_content'].encode('utf-8')
        task = AgenticTaskEnvelope.model_validate_json(task_bytes)
    except ValueError as error:
        raise AgenticAdmissionError(f'model-visible TASK.json is invalid: {error}') from error
    if task_bytes != canonical_json_bytes(task):
        raise AgenticAdmissionError('model-visible TASK.json must use canonical JSON')
    return tuple(files), task


def _require_agentic_audit_identity(workspace: LoadedAgenticWorkspace, audit_input) -> None:
    if (
        audit_input.case_id != workspace.manifest_sha256
        or audit_input.episode_id != workspace.task.episode_id
        or audit_input.decision_package_sha256 != workspace.manifest.workspace_tree_sha256
        or audit_input.episode_manifest_sha256 != workspace.task.episode_manifest_sha256
        or audit_input.public_artifact.artifact_id != AGENTIC_AUDIT_PUBLIC_ARTIFACT_ID
        or audit_input.public_artifact.sha256 != workspace.manifest.model_visible_surface_sha256
        or audit_input.public_artifact.byte_count != len(workspace.model_visible_surface)
    ):
        raise AgenticAdmissionError('contamination audit does not bind the exact Agentic workspace')


def _required_artifact(artifacts: Mapping[str, bytes], key: str) -> bytes:
    value = artifacts.get(key)
    if not isinstance(value, bytes) or not value:
        raise AgenticAdmissionError(f'missing or empty exact-byte contamination artifact: {key}')
    return value


def _require_safe_comparison_ids(payloads: Mapping[str, bytes]) -> None:
    if not payloads:
        raise AgenticAdmissionError('Agentic contamination screening requires a private comparison namespace')
    for artifact_id, payload in payloads.items():
        _require_safe_artifact_id(artifact_id)
        if not isinstance(payload, bytes) or not payload:
            raise AgenticAdmissionError('comparison artifacts must be non-empty exact bytes')


def _require_protected_corpus(
    workspace: LoadedAgenticWorkspace,
    manifest: AgenticProtectedCorpusManifest,
    payloads: Mapping[str, bytes],
) -> None:
    _require_protected_corpus_for_task(workspace.task, manifest, payloads)


def _require_protected_corpus_for_task(
    task: AgenticTaskEnvelope,
    manifest: AgenticProtectedCorpusManifest,
    payloads: Mapping[str, bytes],
) -> None:
    if manifest.historical_cutoff != task.decision_at:
        raise AgenticAdmissionError('protected corpus cutoff does not match the Agentic task cutoff')
    _require_safe_comparison_ids(payloads)
    expected = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
    if set(payloads) != set(expected):
        raise AgenticAdmissionError('comparison payloads do not match the protected corpus inventory')
    for artifact_id, artifact in expected.items():
        payload = payloads[artifact_id]
        if (hashlib.sha256(payload).hexdigest(), len(payload)) != (artifact.sha256, artifact.byte_count):
            raise AgenticAdmissionError('comparison payload does not match its protected corpus binding')


def _require_protected_bindings(manifest: AgenticProtectedCorpusManifest, bindings) -> None:
    expected = tuple((artifact.artifact_id, artifact.sha256, artifact.byte_count) for artifact in manifest.artifacts)
    actual = tuple((artifact.artifact_id, artifact.sha256, artifact.byte_count) for artifact in bindings)
    if actual != expected:
        raise AgenticAdmissionError('audit comparison bindings do not exactly match the protected corpus manifest')


def _require_safe_artifact_id(artifact_id: str) -> None:
    if _SAFE_ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise AgenticAdmissionError('comparison artifact IDs must be lowercase path-safe identifiers')
