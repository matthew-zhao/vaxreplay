from __future__ import annotations

import socket
import struct
import threading
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import vaxreplay.agentic.clinical_guest_bootstrap as bootstrap_module
from tests.test_clinicaltrials_execution_scoring import _case
from vaxreplay.agentic.clinical_guest_bootstrap import (
    CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
    CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
    AuthenticatedClinicalGuestBootstrap,
    ClinicalGuestBootstrapAck,
    ClinicalGuestBootstrapError,
    ClinicalGuestBootstrapFailureCode,
    ClinicalGuestBootstrapHello,
    ClinicalGuestBootstrapReplayGuard,
    ClinicalGuestBootstrapTrustAnchor,
    ClinicalGuestRpcLimits,
    InMemoryClinicalGuestBootstrapReplayGuard,
    SignedClinicalGuestBootstrapHello,
    clinical_guest_bootstrap_authorization_key_id,
    clinical_guest_bootstrap_hello_sha256,
    clinical_guest_bootstrap_receipt_key_id,
    clinical_guest_bootstrap_signed_hello_sha256,
    perform_guest_clinical_bootstrap,
    perform_host_clinical_guest_bootstrap,
    run_lane_a_clinical_guest_entry,
    sign_clinical_guest_bootstrap_hello,
    verify_authenticated_clinical_guest_bootstrap,
    verify_signed_clinical_guest_bootstrap_hello,
)
from vaxreplay.agentic.clinical_guest_harness import (
    LANE_A_GUEST_ACTION_SCHEMA_SHA256,
    LANE_A_GUEST_HARNESS_POLICY_ID,
    LaneAGuestHarnessResult,
)
from vaxreplay.agentic.guest_rpc import (
    GuestRpcClient,
    decode_guest_rpc_frame,
    encode_guest_rpc_frame,
    receive_guest_rpc_frame,
    send_guest_rpc_frame,
)
from vaxreplay.agentic.task_protocol import AgenticTaskInvocation, agentic_task_invocation_sha256
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.operations.signing import LocalEd25519Signer

_NOW = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
_RUN_ID = '1' * 32
_SESSION_ID = '2' * 32
_RECEIPT_KEY = b'bootstrap-receipt-key-material!!'
_AUTHORIZATION_SIGNER = LocalEd25519Signer(Ed25519PrivateKey.from_private_bytes(b'\x11' * 32))
_AUTHORIZATION_PUBLIC_KEY = _AUTHORIZATION_SIGNER.public_key_bytes()
_AUTHORIZATION_KEY_ID = clinical_guest_bootstrap_authorization_key_id(_AUTHORIZATION_PUBLIC_KEY)


def _limits() -> ClinicalGuestRpcLimits:
    return ClinicalGuestRpcLimits(
        maximum_frame_body_bytes=1024 * 1024,
        maximum_session_wire_bytes=8 * 1024 * 1024,
        maximum_requests=100,
        maximum_list_entries=100,
        maximum_read_bytes=32_768,
        maximum_search_results=20,
        maximum_submission_bytes=65_536,
    )


def _hello() -> ClinicalGuestBootstrapHello:
    task, _ = _case(with_fact=True)
    invocation = AgenticTaskInvocation.from_task(task, workspace_manifest_sha256='3' * 64)
    return ClinicalGuestBootstrapHello(
        run_id=_RUN_ID,
        start_redemption_sha256='4' * 64,
        session_id=_SESSION_ID,
        task_invocation=invocation,
        task_invocation_sha256=agentic_task_invocation_sha256(invocation),
        workspace_manifest_sha256='3' * 64,
        workspace_tree_sha256='5' * 64,
        model_visible_surface_sha256='6' * 64,
        execution_policy_sha256='7' * 64,
        worker_bootstrap_profile_sha256='8' * 64,
        worker_spec_sha256='8' * 64,
        harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
        harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        rpc_limits=_limits(),
        nonce='9' * 64,
        valid_from=_NOW - timedelta(seconds=10),
        expires_at=_NOW + timedelta(seconds=50),
    )


def _trust_anchor() -> ClinicalGuestBootstrapTrustAnchor:
    # These are independently fixed image/policy pins, not values copied from a wire hello.
    return ClinicalGuestBootstrapTrustAnchor(
        authorization_key_id=_AUTHORIZATION_KEY_ID,
        ed25519_public_key_hex=_AUTHORIZATION_PUBLIC_KEY.hex(),
        execution_policy_sha256='7' * 64,
        worker_bootstrap_profile_sha256='8' * 64,
        harness_policy_id=LANE_A_GUEST_HARNESS_POLICY_ID,
        harness_policy_sha256=CLINICAL_GUEST_BOOTSTRAP_HARNESS_POLICY_SHA256,
        action_schema_sha256=LANE_A_GUEST_ACTION_SCHEMA_SHA256,
        rpc_limits=_limits(),
    )


def test_trust_anchor_excludes_per_run_wire_values() -> None:
    wire_owned_fields = {
        'run_id',
        'start_redemption_sha256',
        'session_id',
        'task_invocation',
        'task_invocation_sha256',
        'workspace_manifest_sha256',
        'workspace_tree_sha256',
        'model_visible_surface_sha256',
        'nonce',
        'valid_from',
        'expires_at',
    }
    assert wire_owned_fields.isdisjoint(ClinicalGuestBootstrapTrustAnchor.model_fields)

    copied_wire_value = _trust_anchor().model_dump(mode='json') | {'run_id': _RUN_ID}
    with pytest.raises(ValueError):
        ClinicalGuestBootstrapTrustAnchor.model_validate(copied_wire_value)


def _guest_thread(
    connection: socket.socket,
    hello: ClinicalGuestBootstrapHello,
    guard: ClinicalGuestBootstrapReplayGuard,
) -> tuple[threading.Thread, dict[str, object]]:
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome['context'] = perform_guest_clinical_bootstrap(
                connection,
                trust_anchor=_trust_anchor(),
                replay_guard=guard,
                clock=lambda: _NOW,
                timeout_seconds=1,
            )
        except BaseException as error:
            outcome['error'] = error

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def _host_bootstrap(
    connection: socket.socket,
    hello: ClinicalGuestBootstrapHello,
) -> AuthenticatedClinicalGuestBootstrap:
    return perform_host_clinical_guest_bootstrap(
        connection,
        hello=hello,
        authorization_signer=_AUTHORIZATION_SIGNER,
        expected_authorization_key_id=_AUTHORIZATION_KEY_ID,
        receipt_key=_RECEIPT_KEY,
        clock=lambda: _NOW,
        timeout_seconds=1,
    )


def test_socketpair_success_authenticates_a_content_free_receipt() -> None:
    hello = _hello()
    host_socket, guest_socket = socket.socketpair()
    thread, guest_outcome = _guest_thread(
        guest_socket,
        hello,
        InMemoryClinicalGuestBootstrapReplayGuard(),
    )
    try:
        artifact = _host_bootstrap(host_socket, hello)
        thread.join(timeout=2)
    finally:
        host_socket.close()
        guest_socket.close()

    assert not thread.is_alive()
    assert 'error' not in guest_outcome
    receipt = verify_authenticated_clinical_guest_bootstrap(
        artifact,
        key=_RECEIPT_KEY,
        expected_key_id=clinical_guest_bootstrap_receipt_key_id(_RECEIPT_KEY),
        expected_hello=hello,
        trust_anchor=_trust_anchor(),
    )
    assert receipt.hello_sha256 == clinical_guest_bootstrap_hello_sha256(hello)
    assert receipt.signed_hello_sha256 == clinical_guest_bootstrap_signed_hello_sha256(artifact.signed_hello)
    assert receipt.guest_ack_protocol_requires_launcher_signature_verification is True
    assert receipt.guest_signature_verification_remotely_attested is False
    assert receipt.outer_binding_claimed_by_bootstrap_layer is False
    assert receipt.guest_rpc_started_before_ack is False
    receipt_bytes = canonical_json_bytes(receipt)
    assert b'Phase: Phase 1' not in receipt_bytes
    assert b'Caf' not in receipt_bytes


def test_host_rejects_an_ack_not_bound_to_the_signed_authorization() -> None:
    hello = _hello()
    host_socket, guest_socket = socket.socketpair()
    outcome: dict[str, object] = {}

    def tampering_guest() -> None:
        try:
            frame = receive_guest_rpc_frame(
                guest_socket,
                maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
            )
            signed, _ = decode_guest_rpc_frame(
                frame,
                SignedClinicalGuestBootstrapHello,
                maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
            )
            received = signed.hello
            ack = ClinicalGuestBootstrapAck(
                signed_hello_sha256='0' * 64,
                hello_sha256=signed.hello_sha256,
                run_id=received.run_id,
                session_id=received.session_id,
                task_invocation_sha256=received.task_invocation_sha256,
                nonce=received.nonce,
                accepted_at=_NOW,
            )
            send_guest_rpc_frame(
                guest_socket,
                encode_guest_rpc_frame(
                    ack,
                    maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
                ),
            )
        except BaseException as error:
            outcome['error'] = error

    thread = threading.Thread(target=tampering_guest)
    thread.start()
    try:
        with pytest.raises(ClinicalGuestBootstrapError) as raised:
            _host_bootstrap(host_socket, hello)
        thread.join(timeout=2)
    finally:
        host_socket.close()
        guest_socket.close()

    assert not thread.is_alive()
    assert 'error' not in outcome
    assert raised.value.code is ClinicalGuestBootstrapFailureCode.ACK_BINDING_INVALID


def test_guest_rejects_a_tampered_launcher_signature_before_acknowledging() -> None:
    hello = _hello()
    signed = sign_clinical_guest_bootstrap_hello(hello, signer=_AUTHORIZATION_SIGNER)
    replacement = ('0' if signed.signature_hex[0] != '0' else '1') + signed.signature_hex[1:]
    tampered = signed.model_copy(update={'signature_hex': replacement})
    host_socket, guest_socket = socket.socketpair()
    try:
        send_guest_rpc_frame(
            host_socket,
            encode_guest_rpc_frame(
                tampered,
                maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
            ),
        )
        with pytest.raises(ClinicalGuestBootstrapError) as raised:
            perform_guest_clinical_bootstrap(
                guest_socket,
                trust_anchor=_trust_anchor(),
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: _NOW,
                timeout_seconds=1,
            )
    finally:
        host_socket.close()
        guest_socket.close()

    assert raised.value.code is ClinicalGuestBootstrapFailureCode.AUTHORIZATION_INVALID


def test_guest_rejects_static_policy_mismatch_before_acknowledging() -> None:
    hello = _hello()
    signed = sign_clinical_guest_bootstrap_hello(hello, signer=_AUTHORIZATION_SIGNER)
    wrong_anchor = _trust_anchor().model_copy(update={'execution_policy_sha256': 'a' * 64})
    host_socket, guest_socket = socket.socketpair()
    try:
        send_guest_rpc_frame(
            host_socket,
            encode_guest_rpc_frame(
                signed,
                maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
            ),
        )
        with pytest.raises(ClinicalGuestBootstrapError) as raised:
            perform_guest_clinical_bootstrap(
                guest_socket,
                trust_anchor=wrong_anchor,
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: _NOW,
                timeout_seconds=1,
            )
    finally:
        host_socket.close()
        guest_socket.close()

    assert raised.value.code is ClinicalGuestBootstrapFailureCode.TRUST_ANCHOR_MISMATCH


def test_guest_accepts_full_worker_hash_only_under_launcher_signature() -> None:
    """The baked profile is static; the exact config-dependent spec remains signed and retained."""

    hello = _hello().model_copy(update={'worker_spec_sha256': 'f' * 64})
    signed = sign_clinical_guest_bootstrap_hello(hello, signer=_AUTHORIZATION_SIGNER)

    assert (
        verify_signed_clinical_guest_bootstrap_hello(
            signed,
            trust_anchor=_trust_anchor(),
        )
        == hello
    )


def test_guest_rejects_replayed_authorization_on_a_second_socketpair() -> None:
    hello = _hello()
    guard = InMemoryClinicalGuestBootstrapReplayGuard()
    first_host, first_guest = socket.socketpair()
    first_thread, first_outcome = _guest_thread(first_guest, hello, guard)
    try:
        _host_bootstrap(first_host, hello)
        first_thread.join(timeout=2)
    finally:
        first_host.close()
        first_guest.close()
    assert not first_thread.is_alive()
    assert 'error' not in first_outcome

    signed = sign_clinical_guest_bootstrap_hello(hello, signer=_AUTHORIZATION_SIGNER)
    second_host, second_guest = socket.socketpair()
    try:
        send_guest_rpc_frame(
            second_host,
            encode_guest_rpc_frame(
                signed,
                maximum_body_bytes=CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES,
            ),
        )
        with pytest.raises(ClinicalGuestBootstrapError) as raised:
            perform_guest_clinical_bootstrap(
                second_guest,
                trust_anchor=_trust_anchor(),
                replay_guard=guard,
                clock=lambda: _NOW,
                timeout_seconds=1,
            )
    finally:
        second_host.close()
        second_guest.close()

    assert raised.value.code is ClinicalGuestBootstrapFailureCode.REPLAY_REJECTED


def test_guest_rejects_an_oversized_authorization_header() -> None:
    host_socket, guest_socket = socket.socketpair()
    try:
        host_socket.sendall(struct.pack('>I', CLINICAL_GUEST_BOOTSTRAP_MAXIMUM_BODY_BYTES + 1))
        with pytest.raises(ClinicalGuestBootstrapError) as raised:
            perform_guest_clinical_bootstrap(
                guest_socket,
                trust_anchor=_trust_anchor(),
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: _NOW,
                timeout_seconds=1,
            )
    finally:
        host_socket.close()
        guest_socket.close()

    assert raised.value.code is ClinicalGuestBootstrapFailureCode.FRAME_REJECTED


def test_guest_bootstrap_times_out_without_an_authorization() -> None:
    host_socket, guest_socket = socket.socketpair()
    try:
        with pytest.raises(ClinicalGuestBootstrapError) as raised:
            perform_guest_clinical_bootstrap(
                guest_socket,
                trust_anchor=_trust_anchor(),
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: _NOW,
                timeout_seconds=0.05,
            )
    finally:
        host_socket.close()
        guest_socket.close()

    assert raised.value.code is ClinicalGuestBootstrapFailureCode.TIMEOUT


def test_authenticated_receipt_tampering_and_cross_run_substitution_are_rejected() -> None:
    hello = _hello()
    host_socket, guest_socket = socket.socketpair()
    thread, _ = _guest_thread(
        guest_socket,
        hello,
        InMemoryClinicalGuestBootstrapReplayGuard(),
    )
    try:
        artifact = _host_bootstrap(host_socket, hello)
        thread.join(timeout=2)
    finally:
        host_socket.close()
        guest_socket.close()
    tampered = AuthenticatedClinicalGuestBootstrap(
        signed_hello=artifact.signed_hello,
        receipt=artifact.receipt.model_copy(update={'worker_spec_sha256': 'f' * 64}),
        receipt_hmac_sha256=artifact.receipt_hmac_sha256,
    )

    for candidate, expected_hello in (
        (tampered, hello),
        (artifact, hello.model_copy(update={'run_id': 'a' * 32})),
        (artifact, hello.model_copy(update={'worker_spec_sha256': 'e' * 64})),
    ):
        with pytest.raises(ClinicalGuestBootstrapError) as raised:
            verify_authenticated_clinical_guest_bootstrap(
                candidate,
                key=_RECEIPT_KEY,
                expected_key_id=clinical_guest_bootstrap_receipt_key_id(_RECEIPT_KEY),
                expected_hello=expected_hello,
                trust_anchor=_trust_anchor(),
            )
        assert raised.value.code is ClinicalGuestBootstrapFailureCode.RECEIPT_AUTHENTICATION_FAILED


def test_guest_entry_constructs_rpc_client_only_after_ack_on_the_same_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hello = _hello()
    host_socket, guest_socket = socket.socketpair()
    observed: dict[str, object] = {}
    sentinel = cast(LaneAGuestHarnessResult, object())

    class RecordingGuestRpcClient:
        def __init__(
            self,
            connection: socket.socket,
            *,
            session_id: str,
            task_invocation: AgenticTaskInvocation,
            maximum_body_bytes: int,
        ) -> None:
            observed['connection'] = connection
            observed['session_id'] = session_id
            observed['invocation'] = task_invocation
            observed['maximum_body_bytes'] = maximum_body_bytes

    def fake_harness(
        client: GuestRpcClient,
        *,
        task_invocation: AgenticTaskInvocation,
    ) -> LaneAGuestHarnessResult:
        observed['client'] = client
        observed['harness_invocation'] = task_invocation
        return sentinel

    monkeypatch.setattr(bootstrap_module, 'GuestRpcClient', RecordingGuestRpcClient)
    monkeypatch.setattr(bootstrap_module, 'run_lane_a_guest_harness', fake_harness)
    outcome: dict[str, object] = {}

    def run_entry() -> None:
        try:
            outcome['result'] = run_lane_a_clinical_guest_entry(
                guest_socket,
                trust_anchor=_trust_anchor(),
                replay_guard=InMemoryClinicalGuestBootstrapReplayGuard(),
                clock=lambda: _NOW,
                timeout_seconds=1,
            )
        except BaseException as error:
            outcome['error'] = error

    thread = threading.Thread(target=run_entry)
    thread.start()
    try:
        _host_bootstrap(host_socket, hello)
        thread.join(timeout=2)
    finally:
        host_socket.close()
        guest_socket.close()

    assert not thread.is_alive()
    assert 'error' not in outcome
    assert outcome['result'] is sentinel
    assert observed['connection'] is guest_socket
    assert observed['session_id'] == hello.session_id
    assert observed['invocation'] == hello.task_invocation
    assert observed['harness_invocation'] == hello.task_invocation
    assert observed['maximum_body_bytes'] == hello.rpc_limits.maximum_frame_body_bytes
