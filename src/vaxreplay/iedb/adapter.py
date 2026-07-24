"""Content-addressed IEDB snapshot adapter for VaxReplay cohort replay."""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
import secrets
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vaxreplay.bundle import (
    EpisodeBundle,
    body_sha256,
    canonical_json_bytes,
    jsonl_text,
    ranking_labels_commitment,
    records_sha256,
)
from vaxreplay.case_schema import (
    AdapterProvenance,
    AssessmentConclusion,
    CandidateRecord,
    EpisodeManifest,
    EvidenceRecord,
    EvidenceStance,
    ForecastTarget,
    GoldAssessmentRecord,
    GoldEvidenceRecord,
    LabelCommitmentScheme,
    OutcomeRecord,
    PrivateLabels,
    SourceSnapshotCommitment,
    SourceType,
)
from vaxreplay.iedb.raw_schema import (
    IEDB_ADAPTER_ID,
    IEDB_BINARY_RANKING_RUBRIC_VERSION,
    IedbApiMetric,
    IedbAuditSource,
    IedbCandidateMap,
    IedbEndpoint,
    IedbEpisodeSpec,
    IedbEvidenceAudit,
    IedbOutcomeAudit,
    IedbPrivateAudit,
    IedbSnapshotManifest,
    IedbSnapshotTable,
    IedbTableFormat,
    NormalizedIedbAssay,
    QualitativePolarity,
)
from vaxreplay.ranking_schema import RankingLabelV1

_HTML_TAG = re.compile(r'<[^>]+>')
_TARGET_BY_ENDPOINT = {
    IedbEndpoint.TCELL: 'iedb_tcell_qualitative_positive',
    IedbEndpoint.BCELL: 'iedb_bcell_qualitative_positive',
    IedbEndpoint.MHC: 'iedb_mhc_qualitative_positive',
}
_DIMENSION_BY_ENDPOINT = {
    IedbEndpoint.TCELL: 'prior_t_cell_response',
    IedbEndpoint.BCELL: 'prior_b_cell_response',
    IedbEndpoint.MHC: 'prior_mhc_presentation',
}
_MODALITY_LABEL = {
    IedbEndpoint.TCELL: 'T-cell response',
    IedbEndpoint.BCELL: 'B-cell response',
    IedbEndpoint.MHC: 'MHC presentation or binding',
}
_ASSAY_ID_FIELDS = {
    IedbEndpoint.TCELL: ('tcell_iri', 'tcell_id'),
    IedbEndpoint.BCELL: ('bcell_iri', 'bcell_id'),
    IedbEndpoint.MHC: ('elution_iri', 'elution_id'),
}


class IedbAdapterError(ValueError):
    """Raised when a snapshot or episode cannot be normalized safely."""


@dataclass(frozen=True)
class LoadedIedbSnapshot:
    root: Path
    manifest: IedbSnapshotManifest
    manifest_sha256: str
    rows_by_endpoint: dict[IedbEndpoint, tuple[dict[str, Any], ...]]

    def table(self, endpoint: IedbEndpoint) -> IedbSnapshotTable:
        return next(table for table in self.manifest.tables if table.endpoint == endpoint)


@dataclass(frozen=True)
class AssayObservation:
    assay: NormalizedIedbAssay
    normalized_sha256: str
    first_seen_at: datetime
    first_seen_snapshot_id: str
    source_snapshot: LoadedIedbSnapshot
    source_table: IedbSnapshotTable

    @property
    def logical_key(self) -> tuple[IedbEndpoint, str]:
        return (self.assay.endpoint, self.assay.assay_iri)


@dataclass(frozen=True)
class SnapshotHistory:
    snapshots: tuple[LoadedIedbSnapshot, ...]
    states: dict[str, dict[IedbEndpoint, dict[str, AssayObservation]]]
    logical_first_seen: dict[tuple[IedbEndpoint, str], datetime]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise IedbAdapterError(f'cannot read {path}: {error}') from error
    return digest.hexdigest()


def load_snapshot(root: Path) -> LoadedIedbSnapshot:
    root = root.expanduser().resolve()
    try:
        manifest_bytes = (root / 'snapshot.json').read_bytes()
        manifest = IedbSnapshotManifest.model_validate_json(manifest_bytes)
    except OSError as error:
        raise IedbAdapterError(f'cannot read {root / "snapshot.json"}: {error}') from error
    except ValueError as error:
        raise IedbAdapterError(f'invalid IEDB snapshot manifest in {root}: {error}') from error

    metrics_path = _safe_snapshot_path(root, manifest.api_metrics_relative_path)
    metrics_after_path = _safe_snapshot_path(root, manifest.api_metrics_after_relative_path)
    if file_sha256(metrics_path) != manifest.api_metrics_sha256:
        raise IedbAdapterError(f'api_metrics hash mismatch in snapshot {manifest.snapshot_id}')
    if file_sha256(metrics_after_path) != manifest.api_metrics_after_sha256:
        raise IedbAdapterError(f'api_metrics-after hash mismatch in snapshot {manifest.snapshot_id}')
    metrics_before = _load_api_metrics(metrics_path)
    metrics_after = _load_api_metrics(metrics_after_path)
    if metrics_before != metrics_after:
        raise IedbAdapterError(
            f'api_metrics changed during capture of snapshot {manifest.snapshot_id}; retry the capture'
        )
    metric_by_table = {metric.search_table_name: metric for metric in metrics_before}

    rows_by_endpoint: dict[IedbEndpoint, tuple[dict[str, Any], ...]] = {}
    for table in manifest.tables:
        metric = metric_by_table.get(table.endpoint.value)
        if metric is None:
            raise IedbAdapterError(f'api_metrics is missing {table.endpoint.value} in snapshot {manifest.snapshot_id}')
        if _api_metric_timestamp(metric) != manifest.source_build_at:
            raise IedbAdapterError(
                f'api_metrics build timestamp disagrees for {table.endpoint.value} in snapshot {manifest.snapshot_id}'
            )
        if metric.record_count < table.row_count:
            raise IedbAdapterError(
                f'captured {table.row_count} rows but api_metrics reports only {metric.record_count} '
                f'for {table.endpoint.value}'
            )
        table_path = _safe_snapshot_path(root, table.relative_path)
        if file_sha256(table_path) != table.sha256:
            raise IedbAdapterError(f'{table.endpoint.value} file hash mismatch in snapshot {manifest.snapshot_id}')
        try:
            byte_count = table_path.stat().st_size
        except OSError as error:
            raise IedbAdapterError(f'cannot stat {table_path}: {error}') from error
        if byte_count != table.byte_count:
            raise IedbAdapterError(f'{table.endpoint.value} byte count mismatch in snapshot {manifest.snapshot_id}')
        rows = _load_rows(table_path, table.format)
        if len(rows) != table.row_count:
            raise IedbAdapterError(f'{table.endpoint.value} row count mismatch in snapshot {manifest.snapshot_id}')
        if _columns_sha256(rows) != table.columns_sha256:
            raise IedbAdapterError(
                f'{table.endpoint.value} column fingerprint mismatch in snapshot {manifest.snapshot_id}'
            )
        rows_by_endpoint[table.endpoint] = rows

    manifest_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return LoadedIedbSnapshot(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        rows_by_endpoint=rows_by_endpoint,
    )


def load_snapshot_history(
    snapshot_roots: Iterable[Path],
    *,
    required_endpoints: Iterable[IedbEndpoint],
    outcome_as_of: datetime,
) -> SnapshotHistory:
    snapshots = sorted(
        (load_snapshot(root) for root in snapshot_roots),
        key=lambda snapshot: (snapshot.manifest.source_build_at, snapshot.manifest.snapshot_id),
    )
    snapshots = [snapshot for snapshot in snapshots if snapshot.manifest.source_build_at <= outcome_as_of]
    if len(snapshots) < 2:
        raise IedbAdapterError('at least two snapshots at or before outcome_as_of are required')

    snapshot_ids = [snapshot.manifest.snapshot_id for snapshot in snapshots]
    build_times = [snapshot.manifest.source_build_at for snapshot in snapshots]
    if len(snapshot_ids) != len(set(snapshot_ids)):
        raise IedbAdapterError('snapshot IDs must be unique')
    if len(build_times) != len(set(build_times)):
        raise IedbAdapterError('source build timestamps must be unique')
    if len({snapshot.manifest.license_id for snapshot in snapshots}) != 1:
        raise IedbAdapterError('all snapshots for an episode must use the same source license ID')

    endpoint_set = set(required_endpoints)
    for snapshot in snapshots:
        missing = endpoint_set - snapshot.rows_by_endpoint.keys()
        if missing:
            raise IedbAdapterError(
                f'snapshot {snapshot.manifest.snapshot_id} is missing endpoints '
                f'{sorted(endpoint.value for endpoint in missing)}'
            )
    for endpoint in endpoint_set:
        source_urls = {snapshot.table(endpoint).source_url for snapshot in snapshots}
        if len(source_urls) != 1:
            raise IedbAdapterError(f'{endpoint.value} must use the exact same canonical query across snapshot history')
        column_fingerprints = {snapshot.table(endpoint).columns_sha256 for snapshot in snapshots}
        if len(column_fingerprints) != 1:
            raise IedbAdapterError(f'{endpoint.value} column schema changed across snapshot history')

    first_version: dict[tuple[IedbEndpoint, str, str], AssayObservation] = {}
    logical_first_seen: dict[tuple[IedbEndpoint, str], datetime] = {}
    states: dict[str, dict[IedbEndpoint, dict[str, AssayObservation]]] = {}
    for snapshot in snapshots:
        snapshot_state: dict[IedbEndpoint, dict[str, AssayObservation]] = {}
        for endpoint in endpoint_set:
            endpoint_state: dict[str, AssayObservation] = {}
            for raw_row in snapshot.rows_by_endpoint[endpoint]:
                assay = normalize_assay(endpoint, raw_row)
                if assay.assay_iri in endpoint_state:
                    raise IedbAdapterError(
                        f'duplicate logical assay {assay.assay_iri} in snapshot '
                        f'{snapshot.manifest.snapshot_id}/{endpoint.value}'
                    )
                normalized_sha = hashlib.sha256(canonical_json_bytes(assay)).hexdigest()
                version_key = (endpoint, assay.assay_iri, normalized_sha)
                observation = first_version.get(version_key)
                if observation is None:
                    observation = AssayObservation(
                        assay=assay,
                        normalized_sha256=normalized_sha,
                        first_seen_at=snapshot.manifest.source_build_at,
                        first_seen_snapshot_id=snapshot.manifest.snapshot_id,
                        source_snapshot=snapshot,
                        source_table=snapshot.table(endpoint),
                    )
                    first_version[version_key] = observation
                endpoint_state[assay.assay_iri] = observation
                logical_first_seen.setdefault((endpoint, assay.assay_iri), snapshot.manifest.source_build_at)
            snapshot_state[endpoint] = endpoint_state
        states[snapshot.manifest.snapshot_id] = snapshot_state

    return SnapshotHistory(tuple(snapshots), states, logical_first_seen)


def normalize_assay(endpoint: IedbEndpoint, row: dict[str, Any]) -> NormalizedIedbAssay:
    iri_field, id_field = _ASSAY_ID_FIELDS[endpoint]
    assay_iri = _optional_scalar(row.get(iri_field))
    if assay_iri is None:
        assay_id = _optional_scalar(row.get(id_field))
        if assay_id is None:
            raise IedbAdapterError(f'{endpoint.value} row is missing its assay identifier')
        assay_iri = f'IEDB_ASSAY:{assay_id}'

    structure_iri = _optional_scalar(row.get('structure_iri'))
    reference_iri = _optional_scalar(row.get('reference_iri'))
    if structure_iri is None or reference_iri is None:
        raise IedbAdapterError(f'{endpoint.value} assay {assay_iri} requires structure_iri and reference_iri')

    qualitative_measure = _clean_optional(_optional_scalar(row.get('qualitative_measure')))
    curated = _curated_source(row.get('curated_source_antigen'))
    region_start = _optional_int(curated.get('starting_position'))
    region_end = _optional_int(curated.get('ending_position'))
    return NormalizedIedbAssay(
        endpoint=endpoint,
        assay_iri=assay_iri,
        structure_iri=structure_iri,
        qualitative_measure=qualitative_measure,
        polarity=_polarity(qualitative_measure),
        assay_iris=_clean_values(row.get('assay_iris')),
        assay_names=_clean_values(row.get('assay_names')),
        reference_iri=reference_iri,
        pubmed_id=_clean_optional(_optional_scalar(row.get('pubmed_id'))),
        reference_titles=_clean_values(_first_present(row, 'reference_titles', 'reference_title')),
        reference_dates=_clean_values(_first_present(row, 'reference_dates', 'reference_date')),
        parent_source_antigen_iri=_clean_optional(_optional_scalar(row.get('parent_source_antigen_iri'))),
        parent_source_antigen_name=_clean_optional(_optional_scalar(row.get('parent_source_antigen_name'))),
        curated_source_antigen_name=_clean_optional(_optional_scalar(curated.get('name'))),
        region_start=region_start,
        region_end=region_end,
        source_organism_iri=_clean_optional(_optional_scalar(row.get('source_organism_iri'))),
        source_organism_name=_clean_optional(_optional_scalar(row.get('source_organism_name'))),
        host_organism_iri=_clean_optional(_optional_scalar(row.get('host_organism_iri'))),
        host_organism_name=_clean_optional(_optional_scalar(row.get('host_organism_name'))),
        mhc_allele_iri=_clean_optional(_optional_scalar(row.get('mhc_allele_iri'))),
        mhc_allele_name=_clean_optional(_optional_scalar(row.get('mhc_allele_name'))),
    )


def build_episode(
    *,
    spec: IedbEpisodeSpec,
    snapshot_roots: Iterable[Path],
    output_root: Path,
    label_commitment_key: bytes | None = None,
) -> EpisodeBundle:
    if spec.ranking_rubric_version != IEDB_BINARY_RANKING_RUBRIC_VERSION:
        raise IedbAdapterError(
            f'unsupported IEDB ranking rubric {spec.ranking_rubric_version!r}; '
            f'expected {IEDB_BINARY_RANKING_RUBRIC_VERSION!r}'
        )
    commitment_key = label_commitment_key or secrets.token_bytes(32)
    if len(commitment_key) < 32:
        raise IedbAdapterError('label commitment key must contain at least 32 bytes')
    history = load_snapshot_history(
        snapshot_roots,
        required_endpoints=spec.evidence_endpoints,
        outcome_as_of=spec.outcome_as_of,
    )
    decision_snapshot = _boundary_snapshot(history, spec.decision_at, 'decision_at')
    label_snapshot = _boundary_snapshot(history, spec.outcome_as_of, 'outcome_as_of')
    candidate_by_structure = {candidate.structure_iri: candidate.candidate_id for candidate in spec.candidates}

    decision_observations = _candidate_observations(
        history.states[decision_snapshot.manifest.snapshot_id], candidate_by_structure
    )
    visible_candidate_ids = {candidate_id for candidate_id, _ in decision_observations}
    missing_visible = set(candidate_by_structure.values()) - visible_candidate_ids
    if missing_visible:
        raise IedbAdapterError(
            f'every candidate must have decision-snapshot evidence; missing {sorted(missing_visible)}'
        )

    label_observations = _label_observations(
        spec=spec,
        history=history,
        label_snapshot=label_snapshot,
        candidate_by_structure=candidate_by_structure,
    )
    label_by_candidate: dict[str, list[AssayObservation]] = defaultdict(list)
    for candidate_id, observation in label_observations:
        label_by_candidate[candidate_id].append(observation)
    missing_labels = set(candidate_by_structure.values()) - label_by_candidate.keys()
    if missing_labels:
        raise IedbAdapterError(
            f'cohort replay requires a post-cutoff label for every candidate; missing {sorted(missing_labels)}'
        )

    candidates = tuple(
        CandidateRecord(episode_id=spec.episode_id, candidate_id=candidate_id)
        for candidate_id in sorted(candidate_by_structure.values())
    )
    evidence_by_observation: dict[tuple[str, IedbEndpoint, str, str], EvidenceRecord] = {}
    for candidate_id, observation in decision_observations:
        key = (
            candidate_id,
            observation.assay.endpoint,
            observation.assay.assay_iri,
            observation.normalized_sha256,
        )
        evidence_by_observation[key] = _evidence_record(
            spec.episode_id,
            candidate_id,
            observation,
            id_key=commitment_key,
        )
    evidence = tuple(
        sorted(
            evidence_by_observation.values(),
            key=lambda record: (
                record.available_at,
                record.related_candidate_ids,
                record.body_sha256,
                record.evidence_id,
            ),
        )
    )

    target_id = _TARGET_BY_ENDPOINT[spec.label_endpoint]
    horizon_days = int((spec.outcome_as_of - spec.decision_at).total_seconds() // 86_400)
    outcomes: list[OutcomeRecord] = []
    outcome_audits: list[IedbOutcomeAudit] = []
    positive_count = 0
    negative_count = 0
    for candidate_id in sorted(label_by_candidate):
        observations = sorted(
            label_by_candidate[candidate_id],
            key=lambda observation: (
                observation.assay.assay_iri,
                observation.normalized_sha256,
            ),
        )
        polarities = {observation.assay.polarity for observation in observations}
        if QualitativePolarity.UNKNOWN in polarities:
            raise IedbAdapterError(f'candidate {candidate_id} has an unknown future qualitative label')
        if len(polarities) != 1:
            raise IedbAdapterError(f'candidate {candidate_id} has conflicting future qualitative labels')
        outcome = 1 if polarities == {QualitativePolarity.POSITIVE} else 0
        positive_count += outcome
        negative_count += 1 - outcome
        outcomes.append(
            OutcomeRecord(
                episode_id=spec.episode_id,
                candidate_id=candidate_id,
                target_id=target_id,
                horizon_days=horizon_days,
                outcome=outcome,
                candidate_utility=float(outcome),
                revealed_at=spec.outcome_as_of,
            )
        )
        outcome_audits.append(
            IedbOutcomeAudit(
                candidate_id=candidate_id,
                target_id=target_id,
                outcome=outcome,
                sources=[_audit_source(observation) for observation in observations],
            )
        )
    if positive_count < spec.min_positive_candidates:
        raise IedbAdapterError(
            f'episode has {positive_count} positive candidates, below the configured minimum '
            f'{spec.min_positive_candidates}'
        )
    if negative_count < spec.min_negative_candidates:
        raise IedbAdapterError(
            f'episode has {negative_count} negative candidates, below the configured minimum '
            f'{spec.min_negative_candidates}'
        )

    assessments_gold, evidence_gold = _gold_labels(
        spec=spec,
        observations=decision_observations,
        evidence_by_observation=evidence_by_observation,
    )
    if not evidence_gold:
        raise IedbAdapterError('at least one visible positive or negative assay is required for grounding')
    private_labels = PrivateLabels(
        outcomes=outcomes,
        assessments_gold=assessments_gold,
        evidence_gold=evidence_gold,
    )
    ranking_labels = tuple(
        RankingLabelV1(
            episode_id=spec.episode_id,
            candidate_id=outcome.candidate_id,
            relevance_grade=outcome.outcome,
        )
        for outcome in sorted(outcomes, key=lambda outcome: outcome.candidate_id)
    )

    private_audit = IedbPrivateAudit(
        episode_id=spec.episode_id,
        candidate_map=[
            IedbCandidateMap(candidate_id=candidate.candidate_id, structure_iri=candidate.structure_iri)
            for candidate in sorted(spec.candidates, key=lambda candidate: candidate.candidate_id)
        ],
        evidence=[
            IedbEvidenceAudit(
                candidate_id=candidate_id,
                evidence_id=evidence_by_observation[
                    (
                        candidate_id,
                        observation.assay.endpoint,
                        observation.assay.assay_iri,
                        observation.normalized_sha256,
                    )
                ].evidence_id,
                source=_audit_source(observation),
            )
            for candidate_id, observation in decision_observations
        ],
        outcomes=outcome_audits,
    )
    private_audit_sha = hmac.new(commitment_key, canonical_json_bytes(private_audit), hashlib.sha256).hexdigest()
    spec_sha = hmac.new(commitment_key, canonical_json_bytes(spec), hashlib.sha256).hexdigest()
    relevant_snapshots = tuple(
        snapshot for snapshot in history.snapshots if snapshot.manifest.source_build_at <= spec.outcome_as_of
    )
    source_provenance = AdapterProvenance(
        adapter_id=IEDB_ADAPTER_ID,
        episode_spec_commitment=spec_sha,
        decision_snapshot_id=decision_snapshot.manifest.snapshot_id,
        label_snapshot_id=label_snapshot.manifest.snapshot_id,
        snapshot_commitments=[
            SourceSnapshotCommitment(
                snapshot_id=snapshot.manifest.snapshot_id,
                source_build_at=snapshot.manifest.source_build_at,
                manifest_sha256=snapshot.manifest_sha256,
                source_url=snapshot.manifest.source_base_url,
                license_id=snapshot.manifest.license_id,
                license_url=snapshot.manifest.license_url,
                citation=snapshot.manifest.citation,
            )
            for snapshot in relevant_snapshots
        ],
        private_audit_commitment=private_audit_sha,
    )
    required_dimensions = [_DIMENSION_BY_ENDPOINT[endpoint] for endpoint in spec.evidence_endpoints]
    label_commitment_value = ranking_labels_commitment(
        private_labels,
        ranking_labels,
        LabelCommitmentScheme.HMAC_SHA256,
        key=commitment_key,
    )
    manifest = EpisodeManifest(
        episode_id=spec.episode_id,
        lineage_group_id=spec.lineage_group_id,
        synthetic=spec.synthetic,
        split=spec.split,
        decision_at=spec.decision_at,
        portfolio_size=spec.portfolio_size,
        candidate_ids=[candidate.candidate_id for candidate in candidates],
        forecast_targets=[ForecastTarget(target_id=target_id, horizon_days=horizon_days)],
        required_dimensions=required_dimensions,
        evidence_sha256=records_sha256(evidence),
        candidates_sha256=records_sha256(candidates),
        labels_sha256=label_commitment_value,
        label_commitment_scheme=LabelCommitmentScheme.HMAC_SHA256,
        label_commitment_key_id=hashlib.sha256(commitment_key).hexdigest(),
        adjudication_version=(f'{IEDB_ADAPTER_ID}:{spec.ranking_rubric_version}:{spec_sha[:12]}'),
        source_provenance=source_provenance,
        reward_version=spec.reward_version,
    )

    output_root = output_root.expanduser().resolve()
    _write_episode_atomically(
        output_root=output_root,
        manifest=manifest,
        candidates=candidates,
        evidence=evidence,
        private_labels=private_labels,
        ranking_labels=ranking_labels,
        private_audit=private_audit,
        label_commitment_key=commitment_key,
        spec=spec,
    )
    return EpisodeBundle.load(output_root, include_private=True)


def audit_episode(root: Path) -> dict[str, Any]:
    bundle = EpisodeBundle.load(root.expanduser().resolve(), include_private=True)
    provenance = bundle.manifest.source_provenance
    if provenance is None or provenance.adapter_id != IEDB_ADAPTER_ID:
        raise IedbAdapterError('episode is not bound to the IEDB adapter provenance contract')
    audit_path = bundle.root / 'private' / 'iedb_audit.json'
    if audit_path.is_symlink() or not audit_path.is_file():
        raise IedbAdapterError('private IEDB audit must be a regular, non-symlink file')
    try:
        audit = IedbPrivateAudit.model_validate_json(audit_path.read_bytes())
    except OSError as error:
        raise IedbAdapterError(f'cannot read {audit_path}: {error}') from error
    except ValueError as error:
        raise IedbAdapterError(f'invalid private IEDB audit: {error}') from error
    commitment_key = bundle.label_commitment_key
    if commitment_key is None:
        raise IedbAdapterError('private IEDB audit requires the HMAC commitment key')
    spec_path = bundle.root / 'private' / 'iedb_episode_spec.json'
    if spec_path.is_symlink() or not spec_path.is_file():
        raise IedbAdapterError('private IEDB episode spec must be a regular, non-symlink file')
    try:
        spec = IedbEpisodeSpec.model_validate_json(spec_path.read_bytes())
    except OSError as error:
        raise IedbAdapterError(f'cannot read {spec_path}: {error}') from error
    except ValueError as error:
        raise IedbAdapterError(f'invalid private IEDB episode spec: {error}') from error
    spec_sha = hmac.new(commitment_key, canonical_json_bytes(spec), hashlib.sha256).hexdigest()
    if spec_sha != provenance.episode_spec_commitment:
        raise IedbAdapterError('private IEDB episode spec does not match the manifest commitment')
    expected_target = _TARGET_BY_ENDPOINT[spec.label_endpoint]
    expected_horizon_days = int((spec.outcome_as_of - spec.decision_at).total_seconds() // 86_400)
    expected_candidate_ids = sorted(candidate.candidate_id for candidate in spec.candidates)
    if (
        spec.episode_id != bundle.manifest.episode_id
        or spec.lineage_group_id != bundle.manifest.lineage_group_id
        or spec.synthetic != bundle.manifest.synthetic
        or spec.split != bundle.manifest.split
        or spec.decision_at != bundle.manifest.decision_at
        or spec.portfolio_size != bundle.manifest.portfolio_size
        or expected_candidate_ids != bundle.manifest.candidate_ids
        or spec.reward_version != bundle.manifest.reward_version
        or len(bundle.manifest.forecast_targets) != 1
        or bundle.manifest.forecast_targets[0].target_id != expected_target
        or bundle.manifest.forecast_targets[0].horizon_days != expected_horizon_days
    ):
        raise IedbAdapterError('private IEDB episode spec does not reconstruct the episode manifest')
    audit_sha = hmac.new(commitment_key, canonical_json_bytes(audit), hashlib.sha256).hexdigest()
    if audit_sha != provenance.private_audit_commitment:
        raise IedbAdapterError('private IEDB audit hash does not match the manifest commitment')
    if audit.episode_id != bundle.manifest.episode_id:
        raise IedbAdapterError('private IEDB audit episode ID mismatch')
    if sorted(mapping.candidate_id for mapping in audit.candidate_map) != sorted(bundle.manifest.candidate_ids):
        raise IedbAdapterError('private IEDB candidate map does not cover the manifest candidates')
    expected_candidate_map = sorted((candidate.candidate_id, candidate.structure_iri) for candidate in spec.candidates)
    actual_candidate_map = sorted((mapping.candidate_id, mapping.structure_iri) for mapping in audit.candidate_map)
    if actual_candidate_map != expected_candidate_map:
        raise IedbAdapterError('private IEDB candidate map does not match the committed episode spec')
    labels = bundle.private_labels
    assert labels is not None
    outcomes = {(outcome.candidate_id, outcome.target_id): outcome.outcome for outcome in labels.outcomes}
    audit_outcomes = {(outcome.candidate_id, outcome.target_id): outcome.outcome for outcome in audit.outcomes}
    if outcomes != audit_outcomes:
        raise IedbAdapterError('private IEDB audit outcomes do not match the bound labels')
    ranking_labels = bundle.ranking_labels
    if ranking_labels is None:
        raise IedbAdapterError('private IEDB audit requires V1 ranking labels')
    ranking_grades = {label.candidate_id: label.relevance_grade for label in ranking_labels}
    outcome_grades = {outcome.candidate_id: outcome.outcome for outcome in labels.outcomes}
    if ranking_grades != outcome_grades:
        raise IedbAdapterError('V1 ranking labels do not match the IEDB binary outcome rubric')
    if {record.evidence_id for record in audit.evidence} != {record.evidence_id for record in bundle.evidence}:
        raise IedbAdapterError('private IEDB evidence audit does not cover the public evidence')
    return {
        'episode_id': bundle.manifest.episode_id,
        'manifest_sha256': bundle.manifest_sha256,
        'private_audit_commitment': audit_sha,
        'episode_spec_commitment': spec_sha,
        'snapshot_count': len(provenance.snapshot_commitments),
        'candidate_count': len(bundle.manifest.candidate_ids),
        'evidence_count': len(bundle.evidence),
    }


def export_public_episode(root: Path, output_root: Path) -> EpisodeBundle:
    """Create a sealed-test-safe public artifact without private labels, keys, or audits."""
    bundle = EpisodeBundle.load(root.expanduser().resolve(), include_private=False)
    if not bundle.manifest.synthetic:
        raise IedbAdapterError(
            'real IEDB public export is blocked pending a non-reversible evidence representation '
            'and linkage-risk evaluation'
        )
    if len(bundle.evidence) != len(bundle.visible_evidence):
        raise IedbAdapterError('public export refuses bundles containing post-cutoff evidence records')
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise IedbAdapterError(f'output directory already exists: {output_root}')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}.', dir=output_root.parent)).resolve()
    try:
        for filename in ('manifest.json', 'candidates.jsonl', 'evidence.jsonl'):
            source = bundle.root / filename
            if source.is_symlink():
                raise IedbAdapterError(f'public export refuses symbolic link: {source}')
            if not source.is_file():
                raise IedbAdapterError(f'public export is missing required file: {source}')
        (temporary_root / 'manifest.json').write_bytes(canonical_json_bytes(bundle.manifest) + b'\n')
        (temporary_root / 'candidates.jsonl').write_text(jsonl_text(bundle.candidates), encoding='utf-8')
        (temporary_root / 'evidence.jsonl').write_text(jsonl_text(bundle.evidence), encoding='utf-8')
        for filename in ('DATASET_CARD.md', 'ADAPTER_REPORT.json'):
            source = bundle.root / filename
            if source.is_symlink():
                raise IedbAdapterError(f'public export refuses symbolic link: {source}')
        provenance = bundle.manifest.source_provenance
        report = {
            'adapter_id': provenance.adapter_id if provenance is not None else None,
            'episode_id': bundle.manifest.episode_id,
            'candidate_count': len(bundle.manifest.candidate_ids),
            'evidence_count': len(bundle.evidence),
            'visible_evidence_count': len(bundle.visible_evidence),
            'source_provenance': provenance.model_dump(mode='json') if provenance is not None else None,
        }
        (temporary_root / 'ADAPTER_REPORT.json').write_bytes(canonical_json_bytes(report) + b'\n')
        (temporary_root / 'DATASET_CARD.md').write_text(_public_dataset_card(bundle), encoding='utf-8')
        EpisodeBundle.load(temporary_root, include_private=False)
        temporary_root.rename(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return EpisodeBundle.load(output_root, include_private=False)


def _public_dataset_card(bundle: EpisodeBundle) -> str:
    provenance = bundle.manifest.source_provenance
    adapter_id = provenance.adapter_id if provenance is not None else 'unknown'
    return f"""# {bundle.manifest.episode_id}

This is a public VaxReplay episode reconstructed from integrity-checked manifest data.

- Adapter: `{adapter_id}`
- Reward version: `{bundle.manifest.reward_version}`
- Adjudication version: `{bundle.manifest.adjudication_version}`
- Decision time: `{bundle.manifest.decision_at.isoformat()}`
- Candidates: {len(bundle.manifest.candidate_ids)}
- Evidence records: {len(bundle.visible_evidence)}
- Private labels, audits, keys, class counts, and future evidence are excluded.

This cohort-replay artifact evaluates prioritization within a later-assayed cohort. It does not
evaluate discovery of the cohort, vaccine efficacy, clinical protection, or causal development
decisions. Consult the source snapshot commitments in `manifest.json` for attribution and license
metadata.
"""


def _boundary_snapshot(history: SnapshotHistory, timestamp: datetime, field_name: str) -> LoadedIedbSnapshot:
    matches = [snapshot for snapshot in history.snapshots if snapshot.manifest.source_build_at == timestamp]
    if len(matches) != 1:
        raise IedbAdapterError(
            f'{field_name} must exactly equal one pinned snapshot source_build_at; found {len(matches)}'
        )
    return matches[0]


def _candidate_observations(
    state: dict[IedbEndpoint, dict[str, AssayObservation]],
    candidate_by_structure: dict[str, str],
) -> list[tuple[str, AssayObservation]]:
    observations: list[tuple[str, AssayObservation]] = []
    for endpoint in sorted(state, key=lambda value: value.value):
        for observation in state[endpoint].values():
            candidate_id = candidate_by_structure.get(observation.assay.structure_iri)
            if candidate_id is not None:
                observations.append((candidate_id, observation))
    return sorted(
        observations,
        key=lambda item: (
            item[0],
            item[1].assay.endpoint.value,
            item[1].assay.assay_iri,
            item[1].normalized_sha256,
        ),
    )


def _label_observations(
    *,
    spec: IedbEpisodeSpec,
    history: SnapshotHistory,
    label_snapshot: LoadedIedbSnapshot,
    candidate_by_structure: dict[str, str],
) -> list[tuple[str, AssayObservation]]:
    observations: list[tuple[str, AssayObservation]] = []
    label_state = history.states[label_snapshot.manifest.snapshot_id][spec.label_endpoint]
    for observation in label_state.values():
        assay = observation.assay
        candidate_id = candidate_by_structure.get(assay.structure_iri)
        if candidate_id is None or assay.reference_iri != spec.label_reference_iri:
            continue
        if history.logical_first_seen[observation.logical_key] <= spec.decision_at:
            continue
        if spec.label_assay_iri is not None and spec.label_assay_iri not in assay.assay_iris:
            continue
        if assay.mhc_allele_name != spec.label_mhc_restriction:
            continue
        if assay.host_organism_iri != spec.label_host_organism_iri:
            continue
        if assay.source_organism_iri != spec.label_source_organism_iri:
            continue
        observations.append((candidate_id, observation))
    return sorted(
        observations,
        key=lambda item: (item[0], item[1].assay.assay_iri, item[1].normalized_sha256),
    )


def _evidence_record(
    episode_id: str,
    candidate_id: str,
    observation: AssayObservation,
    *,
    id_key: bytes,
) -> EvidenceRecord:
    assay = observation.assay
    evidence_seed = canonical_json_bytes(
        {
            'adapter_id': IEDB_ADAPTER_ID,
            'episode_id': episode_id,
            'candidate_id': candidate_id,
            'endpoint': assay.endpoint,
            'assay_iri': assay.assay_iri,
            'normalized_sha256': observation.normalized_sha256,
        }
    )
    evidence_id = f'ev-{hmac.new(id_key, evidence_seed, hashlib.sha256).hexdigest()[:24]}'
    body = _render_body(candidate_id, assay)
    source_url = urlsplit(observation.source_table.source_url)
    public_provenance_url = urlunsplit((source_url.scheme, source_url.netloc, source_url.path, '', ''))
    return EvidenceRecord(
        episode_id=episode_id,
        evidence_id=evidence_id,
        source_type=SourceType.EXPERIMENTAL,
        collected_at=None,
        available_at=observation.first_seen_at,
        title=f'{_MODALITY_LABEL[assay.endpoint]} evidence for {candidate_id}',
        body=body,
        body_sha256=body_sha256(body),
        related_candidate_ids=[candidate_id],
        provenance_url=public_provenance_url,
        license_id=observation.source_snapshot.manifest.license_id,
        derivation=(
            f'Deterministic {IEDB_ADAPTER_ID} normalization from snapshot '
            f'{observation.first_seen_snapshot_id}. Availability is first observed snapshot time, '
            'not the reported publication year. Row-level provenance is retained in the private audit.'
        ),
    )


def _render_body(candidate_id: str, assay: NormalizedIedbAssay) -> str:
    qualitative = assay.qualitative_measure or 'Not reported'
    lines = [
        f'Candidate: {candidate_id}.',
        f'Assay modality: {_MODALITY_LABEL[assay.endpoint]}.',
        f'Qualitative result: {qualitative}.',
    ]
    _append_values(lines, 'Assay ontology', assay.assay_iris)
    _append_values(lines, 'Assay name', assay.assay_names)
    _append_optional(lines, 'Parent source antigen', assay.parent_source_antigen_name)
    _append_optional(lines, 'Curated source antigen', assay.curated_source_antigen_name)
    if assay.region_start is not None and assay.region_end is not None:
        lines.append(f'Candidate region: positions {assay.region_start}-{assay.region_end}.')
    _append_optional(lines, 'Source organism', assay.source_organism_name)
    _append_optional(lines, 'Host organism', assay.host_organism_name)
    _append_optional(lines, 'MHC restriction', assay.mhc_allele_name)
    _append_values(lines, 'Reference title', assay.reference_titles)
    _append_values(lines, 'Reference date as reported', assay.reference_dates)
    return '\n'.join(lines)


def _gold_labels(
    *,
    spec: IedbEpisodeSpec,
    observations: list[tuple[str, AssayObservation]],
    evidence_by_observation: dict[tuple[str, IedbEndpoint, str, str], EvidenceRecord],
) -> tuple[list[GoldAssessmentRecord], list[GoldEvidenceRecord]]:
    grouped: dict[tuple[str, IedbEndpoint], list[AssayObservation]] = defaultdict(list)
    for candidate_id, observation in observations:
        grouped[(candidate_id, observation.assay.endpoint)].append(observation)

    assessments: list[GoldAssessmentRecord] = []
    gold_evidence: list[GoldEvidenceRecord] = []
    for candidate_id in sorted(candidate.candidate_id for candidate in spec.candidates):
        for endpoint in spec.evidence_endpoints:
            endpoint_observations = sorted(
                grouped[(candidate_id, endpoint)],
                key=lambda observation: (
                    observation.assay.assay_iri,
                    observation.normalized_sha256,
                ),
            )
            known = [
                observation
                for observation in endpoint_observations
                if observation.assay.polarity != QualitativePolarity.UNKNOWN
            ]
            polarities = {observation.assay.polarity for observation in known}
            if not polarities:
                conclusion = AssessmentConclusion.INSUFFICIENT
            elif polarities == {QualitativePolarity.POSITIVE}:
                conclusion = AssessmentConclusion.FAVORABLE
            elif polarities == {QualitativePolarity.NEGATIVE}:
                conclusion = AssessmentConclusion.CONCERN
            else:
                conclusion = AssessmentConclusion.MIXED
            dimension = _DIMENSION_BY_ENDPOINT[endpoint]
            assessments.append(
                GoldAssessmentRecord(
                    episode_id=spec.episode_id,
                    candidate_id=candidate_id,
                    dimension=dimension,
                    conclusion=conclusion,
                )
            )
            for observation in known:
                key = (
                    candidate_id,
                    observation.assay.endpoint,
                    observation.assay.assay_iri,
                    observation.normalized_sha256,
                )
                evidence = evidence_by_observation[key]
                qualitative = observation.assay.qualitative_measure
                assert qualitative is not None
                gold_evidence.append(
                    GoldEvidenceRecord(
                        episode_id=spec.episode_id,
                        candidate_id=candidate_id,
                        dimension=dimension,
                        evidence_id=evidence.evidence_id,
                        stance=(
                            EvidenceStance.SUPPORT
                            if observation.assay.polarity == QualitativePolarity.POSITIVE
                            else EvidenceStance.CONCERN
                        ),
                        quote=f'Qualitative result: {qualitative}.',
                    )
                )
    return assessments, gold_evidence


def _audit_source(observation: AssayObservation) -> IedbAuditSource:
    return IedbAuditSource(
        endpoint=observation.assay.endpoint,
        assay_iri=observation.assay.assay_iri,
        normalized_sha256=observation.normalized_sha256,
        first_seen_at=observation.first_seen_at,
        first_seen_snapshot_id=observation.first_seen_snapshot_id,
        reference_iri=observation.assay.reference_iri,
        pubmed_id=observation.assay.pubmed_id,
    )


def _write_episode_atomically(
    *,
    output_root: Path,
    manifest: EpisodeManifest,
    candidates: tuple[CandidateRecord, ...],
    evidence: tuple[EvidenceRecord, ...],
    private_labels: PrivateLabels,
    ranking_labels: tuple[RankingLabelV1, ...],
    private_audit: IedbPrivateAudit,
    label_commitment_key: bytes,
    spec: IedbEpisodeSpec,
) -> None:
    if output_root.exists():
        raise IedbAdapterError(f'output directory already exists: {output_root}')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}.', dir=output_root.parent)).resolve()
    try:
        private_root = temporary_root / 'private'
        private_root.mkdir()
        (temporary_root / 'manifest.json').write_bytes(canonical_json_bytes(manifest) + b'\n')
        (temporary_root / 'candidates.jsonl').write_text(jsonl_text(candidates), encoding='utf-8')
        (temporary_root / 'evidence.jsonl').write_text(jsonl_text(evidence), encoding='utf-8')
        (private_root / 'outcomes.jsonl').write_text(jsonl_text(private_labels.outcomes), encoding='utf-8')
        (private_root / 'assessments_gold.jsonl').write_text(
            jsonl_text(private_labels.assessments_gold), encoding='utf-8'
        )
        (private_root / 'evidence_gold.jsonl').write_text(jsonl_text(private_labels.evidence_gold), encoding='utf-8')
        (private_root / 'ranking_labels.jsonl').write_text(jsonl_text(ranking_labels), encoding='utf-8')
        (private_root / 'iedb_audit.json').write_bytes(canonical_json_bytes(private_audit) + b'\n')
        (private_root / 'iedb_episode_spec.json').write_bytes(canonical_json_bytes(spec) + b'\n')
        (private_root / 'label_commitment_key.hex').write_text(label_commitment_key.hex() + '\n', encoding='ascii')
        report = {
            'adapter_id': IEDB_ADAPTER_ID,
            'episode_id': spec.episode_id,
            'candidate_count': len(candidates),
            'evidence_count': len(evidence),
            'visible_evidence_count': sum(record.available_at <= spec.decision_at for record in evidence),
            'source_provenance': (
                manifest.source_provenance.model_dump(mode='json') if manifest.source_provenance is not None else None
            ),
        }
        (temporary_root / 'ADAPTER_REPORT.json').write_bytes(canonical_json_bytes(report) + b'\n')
        (temporary_root / 'DATASET_CARD.md').write_text(
            _dataset_card(spec, len(candidates), len(evidence)),
            encoding='utf-8',
        )
        EpisodeBundle.load(temporary_root, include_private=True)
        audit_episode(temporary_root)
        temporary_root.rename(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _dataset_card(
    spec: IedbEpisodeSpec,
    candidate_count: int,
    evidence_count: int,
) -> str:
    synthetic_note = (
        'This episode was generated from fictional IEDB-shaped rows.'
        if spec.synthetic
        else 'This episode was generated from content-addressed IEDB snapshots.'
    )
    return f"""# {spec.episode_id}

{synthetic_note}

- Adapter: `{IEDB_ADAPTER_ID}`
- Reward version: `{spec.reward_version}`
- Ranking rubric: `{spec.ranking_rubric_version}`
- Decision snapshot time: `{spec.decision_at.isoformat()}`
- Label snapshot time: `{spec.outcome_as_of.isoformat()}`
- Candidates: {candidate_count}
- Evidence records: {evidence_count}
- Private qualitative labels and class counts are excluded from public artifacts.

The cohort is future-conditioned: each candidate has a homogeneous assay outcome in the label
snapshot. It evaluates prioritization within that assayed cohort, not discovery of the cohort and not
vaccine efficacy. Publication years are descriptive only. Availability is the first pinned snapshot
where a normalized record version was observed.

IEDB database records are attributed under the source snapshot's declared license. Source papers and
contributed third-party material can retain separate rights; this derivative includes structured
database fields, not paper full text, tables, or figures.

Source documentation:

- https://help.iedb.org/hc/en-us/articles/114094146931-IEDB-Database-Downloads-XML-SQL-CSV-TSV-XLSX-JSON
- https://help.iedb.org/hc/en-us/articles/4402872882189-Immune-Epitope-Database-Query-API-IQ-API
"""


def _safe_snapshot_path(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if path.is_symlink():
        raise IedbAdapterError(f'snapshot files cannot be symbolic links: {relative_path}')
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise IedbAdapterError(f'snapshot path escapes its root: {relative_path}')
    return resolved_path


def _load_rows(path: Path, table_format: IedbTableFormat) -> tuple[dict[str, Any], ...]:
    try:
        if table_format == IedbTableFormat.JSON:
            value = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(value, list):
                raise IedbAdapterError(f'{path} must contain a JSON array')
            raw_rows = value
        else:
            raw_rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    except OSError as error:
        raise IedbAdapterError(f'cannot read {path}: {error}') from error
    except json.JSONDecodeError as error:
        raise IedbAdapterError(f'invalid JSON in {path}: {error}') from error
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict) or not all(isinstance(key, str) for key in row):
            raise IedbAdapterError(f'{path} row {index} must be a JSON object with string keys')
        rows.append(row)
    return tuple(rows)


def _load_api_metrics(path: Path) -> tuple[IedbApiMetric, ...]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        raise IedbAdapterError(f'cannot read {path}: {error}') from error
    except json.JSONDecodeError as error:
        raise IedbAdapterError(f'invalid JSON in {path}: {error}') from error
    if not isinstance(value, list):
        raise IedbAdapterError(f'{path} must contain a JSON array')
    metrics: list[IedbApiMetric] = []
    for index, raw_metric in enumerate(value):
        if not isinstance(raw_metric, dict):
            raise IedbAdapterError(f'{path} metric {index} must be a JSON object')
        try:
            metrics.append(IedbApiMetric.model_validate_json(canonical_json_bytes(raw_metric)))
        except ValueError as error:
            raise IedbAdapterError(f'invalid api_metrics record {index} in {path}: {error}') from error
    table_names = [metric.search_table_name for metric in metrics]
    if len(table_names) != len(set(table_names)):
        raise IedbAdapterError(f'{path} contains duplicate api_metrics table names')
    return tuple(sorted(metrics, key=lambda metric: metric.search_table_name))


def _api_metric_timestamp(metric: IedbApiMetric) -> datetime:
    timestamp = datetime.fromisoformat(metric.creation_date.replace('Z', '+00:00'))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _columns_sha256(rows: Iterable[dict[str, Any]]) -> str:
    columns = sorted({column for row in rows for column in row})
    return hashlib.sha256(canonical_json_bytes(columns)).hexdigest()


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _optional_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise IedbAdapterError('boolean values cannot be normalized as scalar text')
    if isinstance(value, str | int | float):
        text = str(value).strip()
        return text or None
    if isinstance(value, list) and len(value) == 1:
        return _optional_scalar(value[0])
    raise IedbAdapterError(f'expected a scalar value, received {type(value).__name__}')


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise IedbAdapterError('candidate region positions must be integers')
    return value


def _clean_optional(value: str | None, *, max_length: int = 500) -> str | None:
    if value is None:
        return None
    cleaned = html.unescape(_HTML_TAG.sub(' ', value))
    cleaned = ' '.join(cleaned.split())
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 1].rstrip() + '…'
    return cleaned


def _clean_values(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    cleaned: set[str] = set()
    for item in values:
        scalar = _optional_scalar(item)
        text = _clean_optional(scalar)
        if text is not None:
            cleaned.add(text)
    return sorted(cleaned)


def _curated_source(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, list):
        if len(value) != 1:
            raise IedbAdapterError('curated_source_antigen must contain at most one record')
        value = value[0]
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise IedbAdapterError('curated_source_antigen must be an object')
    return value


def _polarity(qualitative_measure: str | None) -> QualitativePolarity:
    if qualitative_measure is None:
        return QualitativePolarity.UNKNOWN
    normalized = qualitative_measure.casefold()
    if normalized in {'positive', 'positive-low', 'positive-intermediate', 'positive-high'}:
        return QualitativePolarity.POSITIVE
    if normalized == 'negative':
        return QualitativePolarity.NEGATIVE
    return QualitativePolarity.UNKNOWN


def _append_optional(lines: list[str], label: str, value: str | None) -> None:
    if value is not None:
        lines.append(f'{label}: {value}.')


def _append_values(lines: list[str], label: str, values: list[str]) -> None:
    if values:
        lines.append(f'{label}: {"; ".join(values)}.')
