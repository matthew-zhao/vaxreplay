from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tests.test_agentic_guest_rpc import (
    _RPC_RECEIPT_KEY,
    _execution_submission,
    _fixture,
)
from tests.test_clinicaltrials_execution_aggregation import _case as _cohort_case
from tests.test_clinicaltrials_execution_aggregation import _manifest as _cohort_manifest
from vaxreplay.agentic.clinical_execution_bridge import (
    ClinicalCollectionFailureCode,
    ClinicalExecutionBridgeError,
    ClinicalExecutionCollectionStatus,
    ClinicalExecutionRunExpectation,
    build_clinical_agentic_workspace,
    clinical_collection_receipt_key_id,
    clinical_workspace_receipt_key_id,
    collect_clinical_execution_sessions,
    load_clinical_agentic_workspace,
    verify_authenticated_clinical_execution_collection,
)
from vaxreplay.agentic.guest_rpc import (
    GuestRpcErrorCode,
    GuestRpcMethod,
    GuestRpcRequest,
    SubmitRequest,
    encode_guest_rpc_frame,
    guest_rpc_session_key_id,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import Split

_WORKSPACE_KEY = b'clinical-public-workspace-test-key-01'
_COLLECTION_KEY = b'clinical-terminal-collection-test-key'


def _workspace(tmp_path: Path, index: int):
    case = _cohort_case(index, split=Split.TEST)
    key_id = clinical_workspace_receipt_key_id(_WORKSPACE_KEY)
    loaded = build_clinical_agentic_workspace(
        task=case.task,
        workspace_id=f'workspace-{index:03d}',
        output_root=tmp_path / f'workspace-{index:03d}',
        receipt_key=_WORKSPACE_KEY,
        expected_receipt_key_id=key_id,
    )
    return case, loaded


def _terminal_session(tmp_path: Path, workspace, *, completed: bool = True):
    fixture = _fixture(
        tmp_path,
        task_invocation=workspace.invocation,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        broker=workspace.brokered_surface(),
    )
    if completed:
        request = GuestRpcRequest(
            session_id=fixture.session.session_id,
            sequence=0,
            method=GuestRpcMethod.SUBMIT.value,
            body=SubmitRequest(submission=_execution_submission(workspace.task)).model_dump(mode='json'),
        )
        fixture.session.handle_frame(
            encode_guest_rpc_frame(
                request,
                maximum_body_bytes=fixture.session.policy.maximum_frame_body_bytes,
            )
        )
    else:
        fixture.session.abort(GuestRpcErrorCode.CONNECTION_CLOSED)
    return fixture.session.seal()


def _run_expectation(workspace, session) -> ClinicalExecutionRunExpectation:
    seal = session.seal
    return ClinicalExecutionRunExpectation(
        episode_id=workspace.task.context.episode_id,
        run_id=seal.run_id,
        attempt_reservation_sha256=seal.attempt_reservation_sha256,
        execution_policy_sha256=seal.execution_policy_sha256,
        worker_spec_sha256=seal.worker_spec_sha256,
        rpc_policy_sha256=seal.rpc_policy_sha256,
        workspace_manifest_sha256=workspace.manifest_sha256,
        workspace_tree_sha256=workspace.manifest.workspace_tree_sha256,
        model_visible_surface_sha256=workspace.manifest.model_visible_surface_sha256,
        task_invocation_sha256=seal.task_invocation_sha256,
        workspace_broker_contract_version=workspace.brokered_surface().contract_version,
        workspace_broker_contract_sha256=workspace.brokered_surface().contract_sha256,
        gateway_capability_id=seal.gateway_capability_id,
        gateway_grant_sha256=seal.gateway_grant_sha256,
        expected_peer_cid=seal.expected_peer_cid,
        rpc_port=seal.rpc_port,
        guest_rpc_receipt_key_id=seal.receipt_key_id,
    )


def _guest_keys() -> dict[str, bytes]:
    return {guest_rpc_session_key_id(_RPC_RECEIPT_KEY): _RPC_RECEIPT_KEY}


def _workspace_keys() -> dict[str, bytes]:
    return {clinical_workspace_receipt_key_id(_WORKSPACE_KEY): _WORKSPACE_KEY}


def test_public_clinical_workspace_is_authenticated_exact_and_broker_only(tmp_path: Path) -> None:
    case, workspace = _workspace(tmp_path, 1)
    paths = {path.relative_to(workspace.root).as_posix() for path in workspace.root.rglob('*') if path.is_file()}

    assert paths == {
        'input/TASK.json',
        'input/TASK.md',
        'input/source-catalog.json',
        'workspace-manifest.json',
        'workspace-receipt.json',
    }
    assert not any(part in {'organizer', 'private'} for path in paths for part in Path(path).parts)
    assert workspace.task == case.task
    assert workspace.invocation.task == case.task
    broker = workspace.brokered_surface()
    assert tuple(item.path for item in broker.list_files()) == ('TASK.json', 'TASK.md', 'source-catalog.json')
    assert broker.read('TASK.json') == canonical_json_bytes(case.task)
    assert broker.search(case.task.context.episode_id)[0].path == 'TASK.json'

    loaded_again = load_clinical_agentic_workspace(
        workspace.root,
        expected_authenticated_receipt_sha256=workspace.authenticated_receipt_sha256,
        receipt_key=_WORKSPACE_KEY,
        expected_receipt_key_id=clinical_workspace_receipt_key_id(_WORKSPACE_KEY),
    )
    assert loaded_again.manifest_sha256 == workspace.manifest_sha256


def test_public_workspace_rejects_extra_files_and_receipt_tampering(tmp_path: Path) -> None:
    _, workspace = _workspace(tmp_path, 1)
    extra = workspace.root / 'input' / 'future.txt'
    extra.write_bytes(b'future outcome')
    extra.chmod(0o600)

    with pytest.raises(ClinicalExecutionBridgeError, match='inventory'):
        load_clinical_agentic_workspace(
            workspace.root,
            expected_authenticated_receipt_sha256=workspace.authenticated_receipt_sha256,
            receipt_key=_WORKSPACE_KEY,
            expected_receipt_key_id=clinical_workspace_receipt_key_id(_WORKSPACE_KEY),
        )

    extra.unlink()
    with pytest.raises(ClinicalExecutionBridgeError, match='external pin'):
        load_clinical_agentic_workspace(
            workspace.root,
            expected_authenticated_receipt_sha256='0' * 64,
            receipt_key=_WORKSPACE_KEY,
            expected_receipt_key_id=clinical_workspace_receipt_key_id(_WORKSPACE_KEY),
        )


@pytest.mark.parametrize('extra_kind', ['empty-directory', 'fifo'])
def test_public_workspace_rejects_every_unexpected_descendant(tmp_path: Path, extra_kind: str) -> None:
    _, workspace = _workspace(tmp_path, 1)
    extra = workspace.root / 'input' / 'unexpected'
    if extra_kind == 'empty-directory':
        extra.mkdir(mode=0o700)
    else:
        os.mkfifo(extra, mode=0o600)

    with pytest.raises(ClinicalExecutionBridgeError, match='inventory|non-regular'):
        load_clinical_agentic_workspace(
            workspace.root,
            expected_authenticated_receipt_sha256=workspace.authenticated_receipt_sha256,
            receipt_key=_WORKSPACE_KEY,
            expected_receipt_key_id=clinical_workspace_receipt_key_id(_WORKSPACE_KEY),
        )


def test_collector_emits_exact_cohort_submission_and_authenticates_retained_sessions(tmp_path: Path) -> None:
    case_1, workspace_1 = _workspace(tmp_path, 1)
    case_2, workspace_2 = _workspace(tmp_path, 2)
    manifest = _cohort_manifest((case_1, case_2))
    session_1 = _terminal_session(tmp_path / 'session-1', workspace_1)
    session_2 = _terminal_session(tmp_path / 'session-2', workspace_2)
    expectations = (
        _run_expectation(workspace_1, session_1),
        _run_expectation(workspace_2, session_2),
    )
    collection_key_id = clinical_collection_receipt_key_id(_COLLECTION_KEY)

    artifact = collect_clinical_execution_sessions(
        manifest=manifest,
        workspaces=(workspace_1, workspace_2),
        run_expectations=expectations,
        sessions=(session_2, session_1),
        workspace_receipt_keys_by_id=_workspace_keys(),
        guest_rpc_receipt_keys_by_id=_guest_keys(),
        receipt_key=_COLLECTION_KEY,
        expected_receipt_key_id=collection_key_id,
    )

    assert artifact.collection.status == ClinicalExecutionCollectionStatus.COMPLETED
    assert artifact.collection.failure_codes == ()
    assert artifact.collection.cohort_submission is not None
    assert tuple(item.episode_id for item in artifact.collection.cohort_submission.submissions) == (
        case_1.task.context.episode_id,
        case_2.task.context.episode_id,
    )
    assert len(artifact.collection.attempts) == 2
    assert all(item.successful_terminal_submission for item in artifact.collection.attempts)
    assert all(item.run_expectation_verified for item in artifact.collection.attempts)
    assert artifact.collection.run_expectations_bound
    assert not artifact.collection.production_run_collector
    assert not artifact.collection.one_attempt_registry_verified
    assert not artifact.collection.provider_identity_verified
    assert not artifact.collection.harness_identity_verified
    assert not artifact.collection.model_route_verified
    verify_authenticated_clinical_execution_collection(
        artifact,
        manifest=manifest,
        workspaces=(workspace_1, workspace_2),
        run_expectations=expectations,
        workspace_receipt_keys_by_id=_workspace_keys(),
        guest_rpc_receipt_keys_by_id=_guest_keys(),
        receipt_key=_COLLECTION_KEY,
        expected_receipt_key_id=collection_key_id,
    )

    tampered = artifact.model_copy(update={'collection_hmac_sha256': hashlib.sha256(b'tampered').hexdigest()})
    with pytest.raises(ClinicalExecutionBridgeError, match='authentication'):
        verify_authenticated_clinical_execution_collection(
            tampered,
            manifest=manifest,
            workspaces=(workspace_1, workspace_2),
            run_expectations=expectations,
            workspace_receipt_keys_by_id=_workspace_keys(),
            guest_rpc_receipt_keys_by_id=_guest_keys(),
            receipt_key=_COLLECTION_KEY,
            expected_receipt_key_id=collection_key_id,
        )


@pytest.mark.parametrize('failure_mode', ['terminal', 'missing', 'duplicate', 'unauthenticated'])
def test_collector_retains_terminal_evidence_and_never_emits_partial_batch(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    case_1, workspace_1 = _workspace(tmp_path, 1)
    case_2, workspace_2 = _workspace(tmp_path, 2)
    manifest = _cohort_manifest((case_1, case_2))
    session_1 = _terminal_session(tmp_path / 'session-1', workspace_1)
    session_2 = _terminal_session(
        tmp_path / 'session-2',
        workspace_2,
        completed=failure_mode != 'terminal',
    )
    sessions = (session_1, session_2)
    expectations = (
        _run_expectation(workspace_1, session_1),
        _run_expectation(workspace_2, session_2),
    )
    keys = _guest_keys()
    expected_code = ClinicalCollectionFailureCode.TASK_TERMINAL_FAILURE
    if failure_mode == 'missing':
        sessions = (session_1,)
        expected_code = ClinicalCollectionFailureCode.MISSING_TASK_ATTEMPT
    elif failure_mode == 'duplicate':
        sessions = (session_1, session_1, session_2)
        expected_code = ClinicalCollectionFailureCode.DUPLICATE_TASK_ATTEMPT
    elif failure_mode == 'unauthenticated':
        keys = {}
        expected_code = ClinicalCollectionFailureCode.UNAUTHENTICATED_ATTEMPT

    artifact = collect_clinical_execution_sessions(
        manifest=manifest,
        workspaces=(workspace_1, workspace_2),
        run_expectations=expectations,
        sessions=sessions,
        workspace_receipt_keys_by_id=_workspace_keys(),
        guest_rpc_receipt_keys_by_id=keys,
        receipt_key=_COLLECTION_KEY,
        expected_receipt_key_id=clinical_collection_receipt_key_id(_COLLECTION_KEY),
    )

    assert artifact.collection.status == ClinicalExecutionCollectionStatus.FAILED
    assert expected_code in artifact.collection.failure_codes
    assert artifact.collection.cohort_submission is None
    assert len(artifact.collection.attempts) == len(sessions)
    assert tuple(item.session for item in artifact.collection.attempts)


def test_collector_fails_closed_when_terminal_session_uses_wrong_workspace_surface(tmp_path: Path) -> None:
    case, workspace = _workspace(tmp_path, 1)
    manifest = _cohort_manifest((case,))
    good_session = _terminal_session(tmp_path / 'good-session', workspace)
    expectation = _run_expectation(workspace, good_session)
    fixture = _fixture(tmp_path / 'wrong-session', task_invocation=workspace.invocation)
    request = GuestRpcRequest(
        session_id=fixture.session.session_id,
        sequence=0,
        method=GuestRpcMethod.SUBMIT.value,
        body=SubmitRequest(submission=_execution_submission(workspace.task)).model_dump(mode='json'),
    )
    fixture.session.handle_frame(
        encode_guest_rpc_frame(request, maximum_body_bytes=fixture.session.policy.maximum_frame_body_bytes)
    )
    wrong_session = fixture.session.seal()

    artifact = collect_clinical_execution_sessions(
        manifest=manifest,
        workspaces=(workspace,),
        run_expectations=(expectation,),
        sessions=(wrong_session,),
        workspace_receipt_keys_by_id=_workspace_keys(),
        guest_rpc_receipt_keys_by_id=_guest_keys(),
        receipt_key=_COLLECTION_KEY,
        expected_receipt_key_id=clinical_collection_receipt_key_id(_COLLECTION_KEY),
    )

    assert artifact.collection.status == ClinicalExecutionCollectionStatus.FAILED
    assert artifact.collection.cohort_submission is None
    assert ClinicalCollectionFailureCode.RUN_EXPECTATION_MISMATCH in artifact.collection.failure_codes
    assert artifact.collection.attempts[0].authenticated
    assert not artifact.collection.attempts[0].run_expectation_verified
