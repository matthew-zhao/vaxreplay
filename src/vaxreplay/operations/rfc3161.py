"""Optional production RFC 3161 provider and offline verifier.

The provider submits a SHA-256 ``MessageImprint`` containing the already-computed
digest of the exact canonical ledger checkpoint.  It never rehashes the digest as a
message.  The verifier does not trust provider metadata: it derives the time, signer,
policy, and imprint from the CMS-authenticated timestamp response under frozen policy
and certificate bytes.

This small verifier deliberately performs no online AIA, OCSP, or CRL retrieval.
``rfc3161-client`` validates the CMS signature, while VaxReplay strictly parses the
signed ``genTime`` and required ``Accuracy``, pins the algorithms and exact TSA path,
and checks that path at the conservative latest possible witness time.  Archival
revocation and ESS signer-attribute validation are outside this V0 and must not be
claimed by callers.

Install the optional ``vaxreplay[witness]`` dependencies to use this module's provider
or verifier.  Importing the models remains safe without the extra installed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import importlib.metadata
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.operations.schema import SAFE_ID_PATTERN, aware_utc
from vaxreplay.operations.witness import (
    AuthenticatedExternalWitnessFacts,
    CheckpointWitnessRequest,
    ExternalWitnessClaim,
    ExternalWitnessMethod,
    WitnessPolicyBinding,
)

RFC3161_AUTHORITY_POLICY_SCHEMA_VERSION = 'vaxreplay.operations-rfc3161-authority-policy.v0.2'
RFC3161_TRUST_POLICY_SCHEMA_VERSION = 'vaxreplay.operations-rfc3161-trust-policy.v0.1'
RFC3161_CERTIFICATE_BINDING_SCHEMA_VERSION = 'vaxreplay.operations-rfc3161-certificate-binding.v0.1'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_OID_PATTERN = r'^[0-2](?:\.(?:0|[1-9][0-9]*)){1,}$'
_SHA256_OID = '2.16.840.1.101.3.4.2.1'
_ECDSA_SHA256_OID = '1.2.840.10045.4.3.2'
_SHA256_ALGORITHM_IDENTIFIER_DER = bytes.fromhex('300b0609608648016503040201')
_SHA256_ALGORITHM_IDENTIFIER_WITH_NULL_DER = bytes.fromhex('300d06096086480165030402010500')
_ECDSA_SHA256_ALGORITHM_IDENTIFIER_DER = bytes.fromhex('300a06082a8648ce3d040302')
_MAX_POLICY_BYTES = 4 * 1024 * 1024
_MAX_CERTIFICATE_DER_BYTES = 1024 * 1024
_MAX_ACCURACY_MICROSECONDS = 86_400_000_000
_DEFAULT_USER_AGENT = 'vaxreplay-rfc3161/0.1'


class Rfc3161Error(ValueError):
    """An RFC 3161 policy, transport, request, or proof failed closed validation."""


class Rfc3161DependencyError(ImportError):
    """The optional RFC 3161 dependency set is unavailable."""


class Rfc3161CertificateBinding(StrictModel):
    """Canonical base64 DER certificate plus its independently visible digest."""

    schema_version: Literal['vaxreplay.operations-rfc3161-certificate-binding.v0.1'] = (
        RFC3161_CERTIFICATE_BINDING_SCHEMA_VERSION
    )
    der_base64: str = Field(min_length=4, max_length=2 * _MAX_CERTIFICATE_DER_BYTES)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode='after')
    def validate_der_binding(self) -> Self:
        der = _decode_canonical_base64(self.der_base64, 'certificate DER')
        if not der or len(der) > _MAX_CERTIFICATE_DER_BYTES:
            raise ValueError('certificate DER has an invalid size')
        if hashlib.sha256(der).hexdigest() != self.sha256:
            raise ValueError('certificate DER does not match its SHA-256 binding')
        return self

    @property
    def der_bytes(self) -> bytes:
        return base64.b64decode(self.der_base64, validate=True)


class Rfc3161AuthorityPolicy(StrictModel):
    """Frozen request, authority, transport, and verifier policy.

    V0.2 is deliberately a narrow Sigstore-compatible profile.  Algorithm OIDs
    and their DER parameter conventions are separate policy fields so accepting
    another RFC-permitted encoding or signature family requires a new explicit
    profile rather than silently widening an existing authority commitment.
    """

    schema_version: Literal['vaxreplay.operations-rfc3161-authority-policy.v0.2'] = (
        RFC3161_AUTHORITY_POLICY_SCHEMA_VERSION
    )
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    endpoint_uri: str = Field(min_length=1, max_length=4096)
    tsa_policy_oid: str = Field(pattern=_OID_PATTERN)
    message_imprint_algorithm: Literal['sha256'] = 'sha256'
    message_imprint_algorithm_parameters: Literal['der-null'] = 'der-null'
    cms_digest_algorithm_oid: Literal['2.16.840.1.101.3.4.2.1'] = _SHA256_OID
    cms_digest_algorithm_parameters: Literal['absent'] = 'absent'
    cms_signature_algorithm_oid: Literal['1.2.840.10045.4.3.2'] = _ECDSA_SHA256_OID
    cms_signature_algorithm_parameters: Literal['absent'] = 'absent'
    accuracy_required: Literal[True] = True
    max_accuracy_microseconds: int = Field(gt=0, le=_MAX_ACCURACY_MICROSECONDS)
    nonce_present: Literal[False] = False
    certificate_requested: Literal[True] = True
    accepted_pki_status: Literal['granted'] = 'granted'
    embedded_certificate_policy: Literal['exact-leaf-and-intermediates'] = 'exact-leaf-and-intermediates'
    authority_valid_from: datetime
    authority_valid_until: datetime | None = None
    timeout_seconds: float = Field(ge=0.1, le=120.0)
    max_response_bytes: int = Field(ge=1, le=16 * 1024 * 1024)
    verifier_id: str = Field(pattern=SAFE_ID_PATTERN)
    verifier_implementation_sha256: str = Field(pattern=_SHA256_PATTERN)
    native_verifier_distribution: Literal['rfc3161-client'] = 'rfc3161-client'
    native_verifier_version: Literal['1.0.7'] = '1.0.7'

    @field_validator('endpoint_uri')
    @classmethod
    def validate_endpoint_uri(cls, value: str) -> str:
        _validate_https_uri(value)
        return value

    @field_validator('authority_valid_from', 'authority_valid_until')
    @classmethod
    def validate_authority_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else aware_utc(value, 'authority validity time')

    @model_validator(mode='after')
    def validate_authority_window(self) -> Self:
        if self.authority_valid_until is not None and self.authority_valid_until <= self.authority_valid_from:
            raise ValueError('authority_valid_until must be later than authority_valid_from')
        return self


class Rfc3161TrustPolicy(StrictModel):
    """Frozen exact certificate path for one RFC 3161 authority."""

    schema_version: Literal['vaxreplay.operations-rfc3161-trust-policy.v0.1'] = RFC3161_TRUST_POLICY_SCHEMA_VERSION
    authority_id: str = Field(pattern=SAFE_ID_PATTERN)
    trust_policy_id: str = Field(pattern=SAFE_ID_PATTERN)
    tsa_certificate: Rfc3161CertificateBinding
    intermediate_certificates: tuple[Rfc3161CertificateBinding, ...] = ()
    trust_anchor_certificate: Rfc3161CertificateBinding
    online_revocation_checking: Literal[False] = False
    archival_revocation_evidence_verified: Literal[False] = False

    @model_validator(mode='after')
    def validate_unique_path(self) -> Self:
        digests = [
            self.tsa_certificate.sha256,
            *(certificate.sha256 for certificate in self.intermediate_certificates),
            self.trust_anchor_certificate.sha256,
        ]
        if len(digests) != len(set(digests)):
            raise ValueError('RFC 3161 trust path certificates must have distinct digests')
        return self


@dataclass(frozen=True)
class Rfc3161TransportRequest:
    """Exact bounded HTTPS request supplied to an injectable transport."""

    endpoint_uri: str
    body: bytes
    timeout_seconds: float
    max_response_bytes: int


@dataclass(frozen=True)
class Rfc3161TransportResponse:
    """HTTP response metadata required for fail-closed provider validation."""

    status_code: int
    content_type: str | None
    body: bytes
    final_uri: str
    content_encoding: str | None = None
    content_length: int | None = None


type Rfc3161Transport = Callable[[Rfc3161TransportRequest], Rfc3161TransportResponse]


@dataclass(frozen=True)
class _StrictTimestampFacts:
    """Security fields obtained from the complete strict ASN.1 parse."""

    embedded_certificate_der: tuple[bytes, ...]
    message_imprint_algorithm_oid: str
    message_imprint: bytes
    tsa_policy_oid: str
    serial_number: int
    gen_time: datetime
    accuracy_components: tuple[int | None, int | None, int | None]
    lower_bound: datetime
    upper_bound: datetime
    nonce: int | None
    signed_data_digest_algorithm_oid: str
    signer_serial_number: int


class Rfc3161CheckpointWitnessProvider:
    """Build and submit a prehashed RFC 3161 request for a checkpoint digest."""

    def __init__(
        self,
        authority_policy_bytes: bytes,
        *,
        transport: Rfc3161Transport | None = None,
    ) -> None:
        self._authority_policy_bytes = _require_exact_bytes(authority_policy_bytes, 'RFC 3161 authority policy')
        self.authority_policy = _load_canonical_model(
            self._authority_policy_bytes,
            Rfc3161AuthorityPolicy,
            'RFC 3161 authority policy',
        )
        _validate_implementation_pin(self.authority_policy)
        self.authority_policy_sha256 = hashlib.sha256(self._authority_policy_bytes).hexdigest()
        self._transport = transport or https_rfc3161_transport

    def __call__(self, request: CheckpointWitnessRequest) -> tuple[ExternalWitnessClaim, bytes]:
        if not isinstance(request, CheckpointWitnessRequest):
            raise Rfc3161Error('RFC 3161 provider requires a CheckpointWitnessRequest')
        policy = self.authority_policy
        if (
            request.authority_id != policy.authority_id
            or request.method is not ExternalWitnessMethod.RFC3161_TIMESTAMP
            or request.policy_id != policy.policy_id
            or request.policy_sha256 != self.authority_policy_sha256
        ):
            raise Rfc3161Error('checkpoint witness request does not match the exact RFC 3161 authority policy bytes')

        request_bytes = build_prehashed_timestamp_request(
            bytes.fromhex(request.checkpoint_sha256),
            tsa_policy_oid=policy.tsa_policy_oid,
        )
        response = self._transport(
            Rfc3161TransportRequest(
                endpoint_uri=policy.endpoint_uri,
                body=request_bytes,
                timeout_seconds=policy.timeout_seconds,
                max_response_bytes=policy.max_response_bytes,
            )
        )
        if not isinstance(response, Rfc3161TransportResponse):
            raise Rfc3161Error('RFC 3161 transport returned an invalid response object')
        if response.status_code != 200:
            raise Rfc3161Error(f'RFC 3161 authority returned HTTP status {response.status_code}')
        if response.final_uri != policy.endpoint_uri:
            raise Rfc3161Error('RFC 3161 transport redirected away from the pinned endpoint')
        if (response.content_type or '').strip().lower() != 'application/timestamp-reply':
            raise Rfc3161Error('RFC 3161 authority returned an invalid Content-Type')
        if response.content_encoding not in (None, '', 'identity'):
            raise Rfc3161Error('RFC 3161 authority returned a content-encoded proof')
        if not isinstance(response.body, bytes) or not response.body:
            raise Rfc3161Error('RFC 3161 authority returned an empty or non-byte proof')
        if len(response.body) > policy.max_response_bytes:
            raise Rfc3161Error('RFC 3161 authority response exceeds the pinned byte limit')
        if response.content_length is not None and response.content_length != len(response.body):
            raise Rfc3161Error('RFC 3161 authority Content-Length does not match the exact response bytes')
        return ExternalWitnessClaim(verification_uri=policy.endpoint_uri), response.body


class Rfc3161CheckpointWitnessVerifier:
    """Offline verifier returning only security facts authenticated from a TSR.

    Certificate status revocation is not checked.  The exact signer and full path are
    instead frozen and hash-bound in ``trust_policy_bytes``; callers needing archival
    OCSP/CRL evidence must add a separate policy layer.
    """

    def __init__(self, authority_policy_bytes: bytes, trust_policy_bytes: bytes) -> None:
        self._authority_policy_bytes = _require_exact_bytes(authority_policy_bytes, 'RFC 3161 authority policy')
        self._trust_policy_bytes = _require_exact_bytes(trust_policy_bytes, 'RFC 3161 trust policy')
        self.authority_policy = _load_canonical_model(
            self._authority_policy_bytes,
            Rfc3161AuthorityPolicy,
            'RFC 3161 authority policy',
        )
        self.trust_policy = _load_canonical_model(
            self._trust_policy_bytes,
            Rfc3161TrustPolicy,
            'RFC 3161 trust policy',
        )
        if self.trust_policy.authority_id != self.authority_policy.authority_id:
            raise Rfc3161Error('RFC 3161 authority and trust policies bind different authorities')
        self.authority_policy_sha256 = hashlib.sha256(self._authority_policy_bytes).hexdigest()
        self.trust_policy_sha256 = hashlib.sha256(self._trust_policy_bytes).hexdigest()
        _validate_implementation_pin(self.authority_policy)

    def __call__(
        self,
        checkpoint_bytes: bytes,
        proof_bytes: bytes,
        expected_policy: WitnessPolicyBinding,
    ) -> AuthenticatedExternalWitnessFacts:
        return self.verify(checkpoint_bytes, proof_bytes, expected_policy)

    def verify(
        self,
        checkpoint_bytes: bytes,
        proof_bytes: bytes,
        expected_policy: WitnessPolicyBinding,
    ) -> AuthenticatedExternalWitnessFacts:
        checkpoint_bytes = _require_exact_bytes(checkpoint_bytes, 'checkpoint')
        proof_bytes = _require_exact_bytes(proof_bytes, 'RFC 3161 proof')
        policy = self.authority_policy
        trust = self.trust_policy
        if len(proof_bytes) > policy.max_response_bytes:
            raise Rfc3161Error('RFC 3161 proof exceeds the pinned byte limit')
        _validate_expected_binding(
            expected_policy,
            authority_policy=policy,
            authority_policy_sha256=self.authority_policy_sha256,
            trust_policy=trust,
            trust_policy_sha256=self.trust_policy_sha256,
        )

        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.serialization import Encoding
            from cryptography.x509 import ObjectIdentifier
            from rfc3161_client import VerifierBuilder, decode_timestamp_response
            from rfc3161_client._rust import verify as vendored_openssl_verify
            from rfc3161_client.errors import VerificationError
        except ImportError as error:  # pragma: no cover - exercised in an environment without the extra
            raise Rfc3161DependencyError('install vaxreplay[witness] to verify RFC 3161 proofs') from error

        tsa_certificate = _load_exact_der_certificate(x509, Encoding, trust.tsa_certificate, 'TSA certificate')
        intermediates = tuple(
            _load_exact_der_certificate(x509, Encoding, binding, 'intermediate certificate')
            for binding in trust.intermediate_certificates
        )
        trust_anchor = _load_exact_der_certificate(
            x509,
            Encoding,
            trust.trust_anchor_certificate,
            'trust anchor certificate',
        )
        _validate_declared_certificate_path(tsa_certificate, intermediates, trust_anchor)
        _validate_timestamping_leaf(x509, tsa_certificate)
        _validate_ca_path(x509, intermediates, trust_anchor)

        strict = _strict_parse_timestamp_response(proof_bytes, policy)
        try:
            timestamp_response = decode_timestamp_response(proof_bytes)
        except Exception as error:
            raise Rfc3161Error(f'invalid RFC 3161 timestamp response: {error}') from error
        try:
            if timestamp_response.as_bytes() != proof_bytes:
                raise Rfc3161Error('RFC 3161 timestamp response is not exact canonical DER')
        except Rfc3161Error:
            raise
        except Exception as error:
            raise Rfc3161Error(f'cannot reproduce exact RFC 3161 response bytes: {error}') from error
        if timestamp_response.status != 0:
            raise Rfc3161Error('RFC 3161 timestamp response status is not GRANTED')

        try:
            tst_info = timestamp_response.tst_info
            imprint = tst_info.message_imprint
            native_imprint_oid = imprint.hash_algorithm.dotted_string
            native_imprint = bytes(imprint.message)
            native_tsa_policy_oid = tst_info.policy.dotted_string
            native_gen_time = aware_utc(tst_info.gen_time, 'RFC 3161 native genTime')
            native_nonce = tst_info.nonce
            native_accuracy = tst_info.accuracy
            native_accuracy_components = (
                None if native_accuracy is None else native_accuracy.seconds,
                None if native_accuracy is None else native_accuracy.millis,
                None if native_accuracy is None else native_accuracy.micros,
            )
            native_tst_info_version = int(tst_info.version)
            native_serial_number = int(tst_info.serial_number)
            native_signed_data_version = int(timestamp_response.signed_data.version)
            native_digest_algorithm_oids = tuple(
                sorted(item.dotted_string for item in timestamp_response.signed_data.digest_algorithms)
            )
            signer_infos = tuple(timestamp_response.signed_data.signer_infos)
            embedded_der = tuple(bytes(item) for item in timestamp_response.signed_data.certificates)
        except Exception as error:
            raise Rfc3161Error(f'cannot parse authenticated RFC 3161 fields: {error}') from error

        if (
            native_tst_info_version != 1
            or native_signed_data_version != 3
            or native_serial_number != strict.serial_number
            or native_gen_time != strict.gen_time
            or native_accuracy_components != strict.accuracy_components
            or native_tsa_policy_oid != strict.tsa_policy_oid
            or native_nonce != strict.nonce
            or native_imprint_oid != strict.message_imprint_algorithm_oid
            or not _constant_time_equal(native_imprint, strict.message_imprint)
            or native_digest_algorithm_oids != (strict.signed_data_digest_algorithm_oid,)
        ):
            raise Rfc3161Error('RFC 3161 strict and native parsers disagree on authenticated security fields')

        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        expected_imprint = bytes.fromhex(checkpoint_sha256)
        if strict.message_imprint_algorithm_oid != _SHA256_OID or len(strict.message_imprint) != 32:
            raise Rfc3161Error('RFC 3161 MessageImprint must use SHA-256 with exactly 32 digest bytes')
        if not _constant_time_equal(strict.message_imprint, expected_imprint):
            raise Rfc3161Error('RFC 3161 MessageImprint binds a different checkpoint SHA-256')
        if strict.tsa_policy_oid != policy.tsa_policy_oid:
            raise Rfc3161Error('RFC 3161 token uses a different TSA policy OID')
        if strict.nonce is not None:
            raise Rfc3161Error('RFC 3161 V0 forbids an unpersisted nonce')
        if strict.lower_bound < policy.authority_valid_from or (
            policy.authority_valid_until is not None and strict.upper_bound >= policy.authority_valid_until
        ):
            raise Rfc3161Error('RFC 3161 authenticated time interval falls outside the authority validity interval')
        if len(signer_infos) != 1:
            raise Rfc3161Error('RFC 3161 timestamp response must contain exactly one SignerInfo')
        if len(embedded_der) != len(strict.embedded_certificate_der) or set(embedded_der) != set(
            strict.embedded_certificate_der
        ):
            raise Rfc3161Error('RFC 3161 parsers disagree on the exact embedded certificate inventory')
        signer_info = signer_infos[0]
        if (
            signer_info.issuer != tsa_certificate.issuer
            or signer_info.serial_number != tsa_certificate.serial_number
            or strict.signer_serial_number != tsa_certificate.serial_number
        ):
            raise Rfc3161Error('RFC 3161 SignerInfo does not identify the pinned TSA certificate')

        expected_embedded_certificates = {
            trust.tsa_certificate.sha256,
            *(binding.sha256 for binding in trust.intermediate_certificates),
        }
        embedded_digests: set[str] = set()
        for certificate_der in embedded_der:
            try:
                certificate = x509.load_der_x509_certificate(certificate_der)
            except Exception as error:
                raise Rfc3161Error(f'RFC 3161 proof contains an invalid embedded certificate: {error}') from error
            if certificate.public_bytes(Encoding.DER) != certificate_der:
                raise Rfc3161Error('RFC 3161 proof contains non-canonical certificate DER')
            digest = hashlib.sha256(certificate_der).hexdigest()
            if digest in embedded_digests:
                raise Rfc3161Error('RFC 3161 proof contains a duplicate embedded certificate')
            embedded_digests.add(digest)
        if embedded_digests != expected_embedded_certificates:
            raise Rfc3161Error('RFC 3161 proof must embed exactly the pinned TSA leaf and intermediates')

        _validate_certificate_interval(
            tsa_certificate,
            strict.lower_bound,
            strict.upper_bound,
            'TSA certificate',
        )
        for index, certificate in enumerate(intermediates):
            _validate_certificate_interval(
                certificate,
                strict.lower_bound,
                strict.upper_bound,
                f'intermediate certificate {index}',
            )
        _validate_certificate_interval(
            trust_anchor,
            strict.lower_bound,
            strict.upper_bound,
            'trust anchor certificate',
        )

        builder = (
            VerifierBuilder()
            .policy_id(ObjectIdentifier(policy.tsa_policy_oid))
            .tsa_certificate(tsa_certificate)
            .add_root_certificate(trust_anchor)
        )
        for certificate in intermediates:
            builder = builder.add_intermediate_certificate(certificate)
        try:
            accepted = builder.build().verify(timestamp_response, expected_imprint)
        except VerificationError as error:
            raise Rfc3161Error(f'RFC 3161 CMS or certificate-path verification failed: {error}') from error
        except Exception as error:
            raise Rfc3161Error(f'RFC 3161 verifier failed: {error}') from error
        if accepted is not True:
            raise Rfc3161Error('RFC 3161 verifier rejected the timestamp response')

        # The public rfc3161-client builder passes all configured certificates to
        # OpenSSL's trust store.  Verify a second time through that package's
        # vendored OpenSSL with *only* the frozen root trusted.  Since the exact leaf
        # and intermediates must be embedded above, they remain untrusted chain
        # material and the CMS path has to terminate at this root.
        try:
            vendored_openssl_verify.pkcs7_verify(
                timestamp_response.time_stamp_token(),
                strict.upper_bound,
                [trust_anchor.public_bytes(Encoding.DER)],
            )
        except Exception as error:
            raise Rfc3161Error(f'RFC 3161 CMS does not build to the pinned trust anchor: {error}') from error

        leaf_sha256 = tsa_certificate.fingerprint(hashes.SHA256()).hex()
        if leaf_sha256 != trust.tsa_certificate.sha256:
            raise Rfc3161Error('authenticated TSA certificate does not match its frozen digest')
        proof_sha256 = hashlib.sha256(proof_bytes).hexdigest()
        return AuthenticatedExternalWitnessFacts(
            receipt_id=f'rfc3161-sha256-{proof_sha256}',
            authority_id=policy.authority_id,
            witness_id=f'x509-sha256-{leaf_sha256}',
            method=ExternalWitnessMethod.RFC3161_TIMESTAMP,
            policy_id=policy.policy_id,
            checkpoint_sha256=checkpoint_sha256,
            witnessed_at=strict.upper_bound,
        )


def make_rfc3161_provider(
    authority_policy_bytes: bytes,
    trust_policy_bytes: bytes,
    *,
    expected_policy: WitnessPolicyBinding,
    transport: Rfc3161Transport | None = None,
) -> Rfc3161CheckpointWitnessProvider:
    """Create a provider only after every generic policy digest is pinned.

    ``expected_policy`` must come from an out-of-band release or deployment
    configuration.  Deriving it from the two mutable policy files at capture time
    would merely prove self-consistency and is not a trust decision.
    """

    authority_bytes, trust_bytes, authority, trust = _load_policy_pair(
        authority_policy_bytes,
        trust_policy_bytes,
    )
    _validate_implementation_pin(authority)
    _validate_expected_binding(
        expected_policy,
        authority_policy=authority,
        authority_policy_sha256=hashlib.sha256(authority_bytes).hexdigest(),
        trust_policy=trust,
        trust_policy_sha256=hashlib.sha256(trust_bytes).hexdigest(),
    )
    return Rfc3161CheckpointWitnessProvider(authority_bytes, transport=transport)


def make_rfc3161_verifier(
    authority_policy_bytes: bytes,
    trust_policy_bytes: bytes,
    *,
    expected_policy: WitnessPolicyBinding,
) -> Rfc3161CheckpointWitnessVerifier:
    """Create the structured-facts verifier under an out-of-band policy pin."""

    authority_bytes, trust_bytes, authority, trust = _load_policy_pair(
        authority_policy_bytes,
        trust_policy_bytes,
    )
    _validate_implementation_pin(authority)
    _validate_expected_binding(
        expected_policy,
        authority_policy=authority,
        authority_policy_sha256=hashlib.sha256(authority_bytes).hexdigest(),
        trust_policy=trust,
        trust_policy_sha256=hashlib.sha256(trust_bytes).hexdigest(),
    )
    return Rfc3161CheckpointWitnessVerifier(authority_bytes, trust_bytes)


def rfc3161_certificate_binding(certificate_der: bytes) -> Rfc3161CertificateBinding:
    """Create the canonical exact-DER certificate binding used in trust policy."""

    der = _require_exact_bytes(certificate_der, 'RFC 3161 certificate DER')
    if len(der) > _MAX_CERTIFICATE_DER_BYTES:
        raise Rfc3161Error('RFC 3161 certificate DER exceeds the configured byte limit')
    return Rfc3161CertificateBinding(
        der_base64=base64.b64encode(der).decode('ascii'),
        sha256=hashlib.sha256(der).hexdigest(),
    )


def rfc3161_verifier_implementation_sha256() -> str:
    """Hash the exact installed Python verifier source for policy pinning.

    This deliberately fails if the source artifact is unavailable; it never replaces
    an implementation commitment with a package version or a caller-provided string.
    The authority policy separately checks native-verifier distribution metadata.
    That version check is not a cryptographic hash of installed dependency code.  A
    trusted deployment must install the reviewed ``uv.lock`` with ``uv sync --locked``
    and protect the resulting runtime as part of its verifier trust boundary.
    """

    try:
        implementation_bytes = Path(__file__).read_bytes()
    except OSError as error:
        raise Rfc3161Error(f'cannot read exact RFC 3161 verifier implementation bytes: {error}') from error
    if not implementation_bytes:
        raise Rfc3161Error('RFC 3161 verifier implementation artifact is empty')
    return hashlib.sha256(implementation_bytes).hexdigest()


def rfc3161_witness_policy_binding(
    authority_policy_bytes: bytes,
    trust_policy_bytes: bytes,
) -> WitnessPolicyBinding:
    """Derive the generic out-of-band binding from exact canonical policy bytes."""

    authority_bytes, trust_bytes, authority, trust = _load_policy_pair(
        authority_policy_bytes,
        trust_policy_bytes,
    )
    _validate_implementation_pin(authority)
    return WitnessPolicyBinding(
        authority_id=authority.authority_id,
        method=ExternalWitnessMethod.RFC3161_TIMESTAMP,
        policy_id=authority.policy_id,
        policy_sha256=hashlib.sha256(authority_bytes).hexdigest(),
        trust_policy_id=trust.trust_policy_id,
        trust_policy_sha256=hashlib.sha256(trust_bytes).hexdigest(),
        verifier_id=authority.verifier_id,
        verifier_implementation_sha256=authority.verifier_implementation_sha256,
    )


def build_prehashed_timestamp_request(checkpoint_sha256: bytes, *, tsa_policy_oid: str) -> bytes:
    """Build DER TSQ whose MessageImprint is the supplied raw checkpoint digest."""

    if not isinstance(checkpoint_sha256, bytes) or len(checkpoint_sha256) != 32:
        raise Rfc3161Error('prehashed RFC 3161 request requires exactly 32 SHA-256 digest bytes')
    if not _valid_oid(tsa_policy_oid):
        raise Rfc3161Error('invalid RFC 3161 TSA policy OID')
    try:
        from asn1crypto import algos, tsp
    except ImportError as error:  # pragma: no cover - exercised in an environment without the extra
        raise Rfc3161DependencyError('install vaxreplay[witness] to build RFC 3161 requests') from error
    try:
        request = tsp.TimeStampReq(
            {
                'version': 'v1',
                'message_imprint': tsp.MessageImprint(
                    {
                        'hash_algorithm': algos.DigestAlgorithm({'algorithm': 'sha256'}),
                        'hashed_message': checkpoint_sha256,
                    }
                ),
                'req_policy': tsa_policy_oid,
                'cert_req': True,
            }
        )
        return request.dump()
    except Exception as error:
        raise Rfc3161Error(f'cannot encode RFC 3161 timestamp request: {error}') from error


def https_rfc3161_transport(request: Rfc3161TransportRequest) -> Rfc3161TransportResponse:
    """POST a bounded TSQ over HTTPS without redirects or content decoding."""

    if not isinstance(request, Rfc3161TransportRequest):
        raise Rfc3161Error('RFC 3161 HTTPS transport requires an Rfc3161TransportRequest')
    _validate_https_uri(request.endpoint_uri)
    if not isinstance(request.body, bytes) or not request.body:
        raise Rfc3161Error('RFC 3161 HTTPS request body must be nonempty bytes')
    if request.max_response_bytes < 1 or request.max_response_bytes > 16 * 1024 * 1024:
        raise Rfc3161Error('RFC 3161 HTTPS response limit is invalid')
    if request.timeout_seconds < 0.1 or request.timeout_seconds > 120:
        raise Rfc3161Error('RFC 3161 HTTPS timeout is invalid')
    try:
        import certifi
    except ImportError as error:  # pragma: no cover - exercised in an environment without the extra
        raise Rfc3161DependencyError('install vaxreplay[witness] for portable HTTPS trust roots') from error

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
            return None

    context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(
        # A witness request carries only a digest, but authority identity and
        # endpoint policy still must not depend on ambient proxy configuration.
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
        _NoRedirect(),
    )
    http_request = urllib.request.Request(
        request.endpoint_uri,
        data=request.body,
        method='POST',
        headers={
            'Accept': 'application/timestamp-reply',
            'Content-Type': 'application/timestamp-query',
            'User-Agent': _DEFAULT_USER_AGENT,
        },
    )
    try:
        with opener.open(http_request, timeout=request.timeout_seconds) as response:
            status = int(response.status)
            final_uri = response.geturl()
            content_type = response.headers.get('Content-Type')
            content_encoding = response.headers.get('Content-Encoding')
            declared_length = response.headers.get('Content-Length')
            if declared_length is not None:
                try:
                    parsed_length = int(declared_length)
                except ValueError as error:
                    raise Rfc3161Error('RFC 3161 authority returned an invalid Content-Length') from error
                if parsed_length < 0 or parsed_length > request.max_response_bytes:
                    raise Rfc3161Error('RFC 3161 authority response exceeds the pinned byte limit')
            body = response.read(request.max_response_bytes + 1)
    except Rfc3161Error:
        raise
    except urllib.error.HTTPError as error:
        raise Rfc3161Error(f'RFC 3161 HTTPS request failed with HTTP status {error.code}') from error
    except (OSError, urllib.error.URLError) as error:
        raise Rfc3161Error(f'RFC 3161 HTTPS request failed: {error}') from error
    if len(body) > request.max_response_bytes:
        raise Rfc3161Error('RFC 3161 authority response exceeds the pinned byte limit')
    return Rfc3161TransportResponse(
        status_code=status,
        content_type=content_type,
        body=body,
        final_uri=final_uri,
        content_encoding=content_encoding,
        content_length=parsed_length if declared_length is not None else None,
    )


def _load_policy_pair(
    authority_policy_bytes: bytes,
    trust_policy_bytes: bytes,
) -> tuple[bytes, bytes, Rfc3161AuthorityPolicy, Rfc3161TrustPolicy]:
    authority_bytes = _require_exact_bytes(authority_policy_bytes, 'RFC 3161 authority policy')
    trust_bytes = _require_exact_bytes(trust_policy_bytes, 'RFC 3161 trust policy')
    authority = _load_canonical_model(
        authority_bytes,
        Rfc3161AuthorityPolicy,
        'RFC 3161 authority policy',
    )
    trust = _load_canonical_model(trust_bytes, Rfc3161TrustPolicy, 'RFC 3161 trust policy')
    if authority.authority_id != trust.authority_id:
        raise Rfc3161Error('RFC 3161 authority and trust policies bind different authorities')
    return authority_bytes, trust_bytes, authority, trust


def _validate_implementation_pin(authority_policy: Rfc3161AuthorityPolicy) -> None:
    actual = rfc3161_verifier_implementation_sha256()
    if not hmac.compare_digest(authority_policy.verifier_implementation_sha256, actual):
        raise Rfc3161Error('RFC 3161 authority policy does not pin the exact installed verifier implementation')
    try:
        native_version = importlib.metadata.version(authority_policy.native_verifier_distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise Rfc3161DependencyError(
            'install the authority-policy-pinned rfc3161-client native verifier dependency'
        ) from error
    if native_version != authority_policy.native_verifier_version:
        raise Rfc3161Error(
            'RFC 3161 authority policy native verifier version does not match the installed distribution'
        )


def _strict_parse_timestamp_response(
    proof_bytes: bytes,
    policy: Rfc3161AuthorityPolicy,
) -> _StrictTimestampFacts:
    """Parse canonical DER and obtain every security field from strict ASN.1.

    rfc3161-client exposes certificates and signer infos as Python sets, which is
    convenient but would hide duplicate ASN.1 elements.  This independent strict
    parse rejects trailing bytes, preserves cardinality, and computes the complete
    RFC 3161 uncertainty interval before the native verifier is invoked.
    """

    try:
        from asn1crypto import tsp
    except ImportError as error:  # pragma: no cover - exercised without the optional extra
        raise Rfc3161DependencyError('install vaxreplay[witness] to parse RFC 3161 proofs') from error
    try:
        response = tsp.TimeStampResp.load(proof_bytes, strict=True)
        if response.dump() != proof_bytes:
            raise Rfc3161Error('RFC 3161 timestamp response is not exact canonical DER')
        if response['status']['status'].native != 'granted':
            raise Rfc3161Error('RFC 3161 timestamp response status is not GRANTED')
        token = response['time_stamp_token']
        if token.native is None or token['content_type'].native != 'signed_data':
            raise Rfc3161Error('RFC 3161 timestamp response lacks a SignedData token')
        signed_data = token['content']
        if signed_data['version'].native != 'v3':
            raise Rfc3161Error('RFC 3161 SignedData must use version 3')
        digest_algorithms = signed_data['digest_algorithms']
        if len(digest_algorithms) != 1:
            raise Rfc3161Error('RFC 3161 SignedData must declare exactly one digest algorithm')
        signed_data_digest_algorithm = digest_algorithms[0]
        signed_data_digest_algorithm_oid = signed_data_digest_algorithm['algorithm'].dotted
        if (
            signed_data_digest_algorithm_oid != policy.cms_digest_algorithm_oid
            or signed_data_digest_algorithm.dump() != _SHA256_ALGORITHM_IDENTIFIER_DER
        ):
            raise Rfc3161Error('RFC 3161 SignedData digest algorithm does not match the pinned SHA-256 policy')

        encapsulated = signed_data['encap_content_info']
        if encapsulated['content_type'].native != 'tst_info' or encapsulated['content'].native is None:
            raise Rfc3161Error('RFC 3161 SignedData does not encapsulate TSTInfo')
        tst_info = encapsulated['content'].parsed
        if not isinstance(tst_info, tsp.TSTInfo):
            raise Rfc3161Error('RFC 3161 encapsulated content is not TSTInfo')
        tst_info.dump()
        if tst_info['version'].native != 'v1':
            raise Rfc3161Error('RFC 3161 TSTInfo must use version 1')

        message_imprint = tst_info['message_imprint']
        message_imprint_algorithm = message_imprint['hash_algorithm']
        message_imprint_algorithm_oid = message_imprint_algorithm['algorithm'].dotted
        if (
            message_imprint_algorithm_oid != _SHA256_OID
            or message_imprint_algorithm.dump() != _SHA256_ALGORITHM_IDENTIFIER_WITH_NULL_DER
        ):
            raise Rfc3161Error('RFC 3161 MessageImprint algorithm does not match the pinned SHA-256 policy')
        authenticated_imprint = message_imprint['hashed_message'].native
        if not isinstance(authenticated_imprint, bytes):
            raise Rfc3161Error('RFC 3161 MessageImprint digest is not exact bytes')

        tsa_policy_oid = tst_info['policy'].dotted
        serial_number = tst_info['serial_number'].native
        if not isinstance(serial_number, int) or serial_number <= 0:
            raise Rfc3161Error('RFC 3161 TSTInfo serialNumber must be a positive integer')

        encoded_gen_time = tst_info['gen_time'].contents
        if len(encoded_gen_time) != 15 or not encoded_gen_time[:14].isdigit() or encoded_gen_time[-1:] != b'Z':
            raise Rfc3161Error('RFC 3161 V0 genTime must be UTC with whole-second precision')
        gen_time = aware_utc(tst_info['gen_time'].native, 'RFC 3161 strict genTime')
        if gen_time.microsecond != 0:
            raise Rfc3161Error('RFC 3161 V0 rejects fractional genTime')

        accuracy = tst_info['accuracy']
        if accuracy.native is None:
            raise Rfc3161Error('RFC 3161 authority policy requires a signed Accuracy field')
        seconds = accuracy['seconds'].native
        millis = accuracy['millis'].native
        micros = accuracy['micros'].native
        if seconds is not None and (not isinstance(seconds, int) or seconds < 0):
            raise Rfc3161Error('RFC 3161 Accuracy seconds must be a nonnegative integer')
        if millis is not None and (not isinstance(millis, int) or not 1 <= millis <= 999):
            raise Rfc3161Error('RFC 3161 Accuracy millis must be in the range 1..999')
        if micros is not None and (not isinstance(micros, int) or not 1 <= micros <= 999):
            raise Rfc3161Error('RFC 3161 Accuracy micros must be in the range 1..999')
        accuracy_microseconds = (seconds or 0) * 1_000_000 + (millis or 0) * 1_000 + (micros or 0)
        if accuracy_microseconds <= 0:
            raise Rfc3161Error('RFC 3161 signed Accuracy must define a positive uncertainty interval')
        if accuracy_microseconds > policy.max_accuracy_microseconds:
            raise Rfc3161Error('RFC 3161 signed Accuracy exceeds the authority-policy maximum')
        accuracy_delta = timedelta(microseconds=accuracy_microseconds)

        if tst_info['extensions'].native is not None:
            raise Rfc3161Error('RFC 3161 V0 rejects TSTInfo extensions')
        nonce = tst_info['nonce'].native
        if nonce is not None and not isinstance(nonce, int):
            raise Rfc3161Error('RFC 3161 TSTInfo nonce is not an integer')

        signer_infos = signed_data['signer_infos']
        if len(signer_infos) != 1:
            raise Rfc3161Error('RFC 3161 timestamp response must contain exactly one SignerInfo')
        signer_info = signer_infos[0]
        if signer_info['version'].native != 'v1' or signer_info['sid'].name != 'issuer_and_serial_number':
            raise Rfc3161Error('RFC 3161 SignerInfo must use version 1 with issuerAndSerialNumber')
        signer_digest_algorithm = signer_info['digest_algorithm']
        if (
            signer_digest_algorithm.dump() != signed_data_digest_algorithm.dump()
            or signer_digest_algorithm['algorithm'].dotted != signed_data_digest_algorithm_oid
        ):
            raise Rfc3161Error('RFC 3161 SignerInfo digest algorithm is inconsistent with SignedData')
        signer_signature_algorithm = signer_info['signature_algorithm']
        if (
            signer_signature_algorithm['algorithm'].dotted != policy.cms_signature_algorithm_oid
            or signer_signature_algorithm.dump() != _ECDSA_SHA256_ALGORITHM_IDENTIFIER_DER
        ):
            raise Rfc3161Error(
                'RFC 3161 SignerInfo signature algorithm does not match the pinned ECDSA-with-SHA256 policy'
            )
        signer_serial_number = signer_info['sid'].chosen['serial_number'].native
        if not isinstance(signer_serial_number, int) or signer_serial_number <= 0:
            raise Rfc3161Error('RFC 3161 SignerInfo serialNumber must be a positive integer')

        certificate_set = signed_data['certificates']
        embedded: list[bytes] = []
        if certificate_set.native is not None:
            for choice in certificate_set:
                if choice.name != 'certificate':
                    raise Rfc3161Error('RFC 3161 proof embeds a non-X.509 certificate choice')
                embedded.append(choice.chosen.dump())
        return _StrictTimestampFacts(
            embedded_certificate_der=tuple(embedded),
            message_imprint_algorithm_oid=message_imprint_algorithm_oid,
            message_imprint=authenticated_imprint,
            tsa_policy_oid=tsa_policy_oid,
            serial_number=serial_number,
            gen_time=gen_time,
            accuracy_components=(seconds, millis, micros),
            lower_bound=gen_time - accuracy_delta,
            upper_bound=gen_time + accuracy_delta,
            nonce=nonce,
            signed_data_digest_algorithm_oid=signed_data_digest_algorithm_oid,
            signer_serial_number=signer_serial_number,
        )
    except Rfc3161Error:
        raise
    except Exception as error:
        raise Rfc3161Error(f'invalid complete RFC 3161 timestamp response: {error}') from error


def _validate_expected_binding(
    expected: WitnessPolicyBinding,
    *,
    authority_policy: Rfc3161AuthorityPolicy,
    authority_policy_sha256: str,
    trust_policy: Rfc3161TrustPolicy,
    trust_policy_sha256: str,
) -> None:
    if not isinstance(expected, WitnessPolicyBinding):
        raise Rfc3161Error('RFC 3161 verifier requires a WitnessPolicyBinding')
    if (
        expected.authority_id != authority_policy.authority_id
        or expected.method is not ExternalWitnessMethod.RFC3161_TIMESTAMP
        or expected.policy_id != authority_policy.policy_id
        or expected.policy_sha256 != authority_policy_sha256
        or expected.trust_policy_id != trust_policy.trust_policy_id
        or expected.trust_policy_sha256 != trust_policy_sha256
        or expected.verifier_id != authority_policy.verifier_id
        or expected.verifier_implementation_sha256 != authority_policy.verifier_implementation_sha256
    ):
        raise Rfc3161Error('expected witness policy does not match the exact RFC 3161 policy and trust bytes')


def _load_exact_der_certificate(x509, encoding, binding, label):  # noqa: ANN001, ANN201
    der = binding.der_bytes
    try:
        certificate = x509.load_der_x509_certificate(der)
    except Exception as error:
        raise Rfc3161Error(f'invalid {label} DER: {error}') from error
    if certificate.public_bytes(encoding.DER) != der:
        raise Rfc3161Error(f'{label} must use exact canonical DER')
    return certificate


def _validate_declared_certificate_path(tsa_certificate, intermediates, trust_anchor) -> None:  # noqa: ANN001
    path = (tsa_certificate, *intermediates, trust_anchor)
    if trust_anchor.subject != trust_anchor.issuer:
        raise Rfc3161Error('RFC 3161 trust anchor must be self-issued')
    for child, issuer in zip(path, path[1:]):
        if child.issuer != issuer.subject:
            raise Rfc3161Error('RFC 3161 certificate bindings are not an ordered issuer path')


def _validate_timestamping_leaf(x509, certificate) -> None:  # noqa: ANN001
    try:
        basic_constraints = certificate.extensions.get_extension_for_class(x509.BasicConstraints)
    except x509.ExtensionNotFound:
        basic_constraints = None
    if basic_constraints is not None and basic_constraints.value.ca:
        raise Rfc3161Error('pinned TSA certificate must be an end-entity certificate')
    try:
        key_usage = certificate.extensions.get_extension_for_class(x509.KeyUsage)
    except x509.ExtensionNotFound as error:
        raise Rfc3161Error('pinned TSA certificate lacks a KeyUsage extension') from error
    if not key_usage.critical or not (key_usage.value.digital_signature or key_usage.value.content_commitment):
        raise Rfc3161Error('pinned TSA critical KeyUsage must permit signing')
    try:
        extension = certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    except x509.ExtensionNotFound as error:
        raise Rfc3161Error('pinned TSA certificate lacks an ExtendedKeyUsage extension') from error
    if not extension.critical:
        raise Rfc3161Error('pinned TSA ExtendedKeyUsage extension must be critical')
    purposes = tuple(extension.value)
    if len(purposes) != 1 or purposes[0] != x509.ExtendedKeyUsageOID.TIME_STAMPING:
        raise Rfc3161Error('pinned TSA certificate EKU must contain only id-kp-timeStamping')


def _validate_ca_path(x509, intermediates, trust_anchor) -> None:  # noqa: ANN001
    ca_certificates = (*intermediates, trust_anchor)
    for index, certificate in enumerate(ca_certificates):
        label = 'trust anchor' if index == len(ca_certificates) - 1 else f'intermediate certificate {index}'
        try:
            basic_constraints = certificate.extensions.get_extension_for_class(x509.BasicConstraints)
        except x509.ExtensionNotFound as error:
            raise Rfc3161Error(f'RFC 3161 {label} lacks BasicConstraints') from error
        if not basic_constraints.critical or not basic_constraints.value.ca:
            raise Rfc3161Error(f'RFC 3161 {label} must have critical CA BasicConstraints')
        subordinate_ca_count = index
        path_length = basic_constraints.value.path_length
        if path_length is not None and path_length < subordinate_ca_count:
            raise Rfc3161Error(f'RFC 3161 {label} BasicConstraints path length is too short')
        try:
            key_usage = certificate.extensions.get_extension_for_class(x509.KeyUsage)
        except x509.ExtensionNotFound as error:
            raise Rfc3161Error(f'RFC 3161 {label} lacks KeyUsage') from error
        if not key_usage.critical or not key_usage.value.key_cert_sign:
            raise Rfc3161Error(f'RFC 3161 {label} critical KeyUsage must permit certificate signing')


def _validate_certificate_interval(  # noqa: ANN001
    certificate,
    lower_bound: datetime,
    upper_bound: datetime,
    label: str,
) -> None:
    not_before = certificate.not_valid_before_utc
    not_after = certificate.not_valid_after_utc
    if lower_bound < not_before or upper_bound >= not_after:
        raise Rfc3161Error(f'{label} was not valid throughout the authenticated RFC 3161 time interval')


def _load_canonical_model[ModelT: StrictModel](
    exact_bytes: bytes,
    model: type[ModelT],
    label: str,
) -> ModelT:
    if len(exact_bytes) > _MAX_POLICY_BYTES:
        raise Rfc3161Error(f'{label} exceeds the configured byte limit')
    try:
        value = model.model_validate_json(exact_bytes)
    except ValueError as error:
        raise Rfc3161Error(f'invalid {label}: {error}') from error
    if canonical_json_bytes(value) != exact_bytes:
        raise Rfc3161Error(f'{label} must use exact canonical JSON encoding')
    return value


def _require_exact_bytes(value: bytes, label: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise Rfc3161Error(f'{label} must be nonempty exact bytes')
    return bytes(value)


def _decode_canonical_base64(value: str, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f'{label} must use canonical base64') from error
    if base64.b64encode(decoded).decode('ascii') != value:
        raise ValueError(f'{label} must use canonical base64')
    return decoded


def _valid_oid(value: str) -> bool:
    if not isinstance(value, str):
        return False
    pieces = value.split('.')
    if len(pieces) < 2 or any(not piece.isdigit() or (len(piece) > 1 and piece[0] == '0') for piece in pieces):
        return False
    first, second = int(pieces[0]), int(pieces[1])
    return first in (0, 1, 2) and (first == 2 or second <= 39)


def _validate_https_uri(value: str) -> None:
    if value.strip() != value or any(character in value for character in '\x00\r\n'):
        raise ValueError('RFC 3161 endpoint URI must be trimmed and contain no control separators')
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as error:
        raise ValueError('RFC 3161 endpoint URI is invalid') from error
    if (
        parts.scheme.lower() != 'https'
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or port == 0
    ):
        raise ValueError('RFC 3161 endpoint URI must be an HTTPS URL without credentials or a fragment')


def _constant_time_equal(left: bytes, right: bytes) -> bool:
    return hmac.compare_digest(left, right)
