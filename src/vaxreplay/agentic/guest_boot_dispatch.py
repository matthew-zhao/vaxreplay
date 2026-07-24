"""Versioned, non-circular boot identity for one Lane A guest executable.

The dispatch manifest contains only facts that are knowable before the task disks are built.  It
does not contain the resulting harness-image or build-receipt digest, because either value would
make the image construction circular.  The externally pinned submitted-harness and canonical
operator manifests bind those final identities back to this manifest and its receipt.

This module describes and validates dispatch.  It does not claim that an artifact was built or
qualified on Linux/KVM.  The native appliance is runtime-integrated; Codex and Claude Code are
limited to development packaging until separate adapter and Linux/KVM qualification exists.
Cursor and custom harnesses have no bootable implementation at this boundary.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from vaxreplay.agentic.submitted_harness import (
    HarnessFamily,
    HarnessRuntimeSupport,
    SubmittedHarnessManifest,
)
from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel

GUEST_BOOT_DISPATCH_MANIFEST_SCHEMA_VERSION = 'vaxreplay.guest-boot-dispatch-manifest.dev-v0.1'
GUEST_BOOT_DISPATCH_ID = 'vaxreplay-lane-a-guest-boot-dispatch'
GUEST_CONFIG_DIGEST_FLAG = '--expected-config-sha256'
NATIVE_GUEST_EXECUTABLE_PATH = '/opt/vaxreplay/bin/vaxreplay-lane-a-clinical-guest'
NATIVE_GUEST_CONFIG_PATH = '/opt/vaxreplay/etc/lane-a-clinical-guest.json'
HEADLESS_GUEST_EXECUTABLE_PATH = '/opt/vaxreplay/bin/vaxreplay-headless-guest-adapter'
HEADLESS_GUEST_CONFIG_PATH = '/opt/vaxreplay/etc/headless-guest-adapter.json'

_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_SAFE_DISPATCH_ARGUMENT_PATTERN = r'^[A-Za-z0-9_./:@+,=-]+$'
_MAXIMUM_MANIFEST_BYTES = 64 * 1024
_MAXIMUM_CONFIG_BYTES = 1024 * 1024


class GuestBootDispatchError(ValueError):
    """A dispatch manifest, config, or submitted-harness binding failed closed."""


class GuestBootConfigSchema(str, enum.Enum):
    LANE_A_CLINICAL = 'lane_a_clinical_guest_dev_v0.2'
    HEADLESS_ADAPTER = 'headless_guest_adapter_dev_v0.1'


class GuestBootDispatchAdmission(str, enum.Enum):
    RUNTIME_INTEGRATED_REQUIRES_EXTERNAL_QUALIFICATION = 'runtime_integrated_requires_external_qualification'
    DEVELOPMENT_PACKAGING_ONLY = 'development_packaging_only'


class GuestBootDispatchManifest(StrictModel):
    """Exact pre-build identity and argv for one guest executable and config."""

    schema_version: Literal['vaxreplay.guest-boot-dispatch-manifest.dev-v0.1'] = (
        GUEST_BOOT_DISPATCH_MANIFEST_SCHEMA_VERSION
    )
    dispatch_id: Literal['vaxreplay-lane-a-guest-boot-dispatch'] = GUEST_BOOT_DISPATCH_ID
    family: HarnessFamily
    runtime_support: HarnessRuntimeSupport
    admission: GuestBootDispatchAdmission
    config_schema: GuestBootConfigSchema
    guest_executable_path: str = Field(min_length=2, max_length=4096)
    guest_executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_config_path: str = Field(min_length=2, max_length=4096)
    guest_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    guest_argv: tuple[str, ...] = Field(min_length=3, max_length=3)
    guest_environment: tuple[str, ...] = Field(default=(), max_length=0)
    pid1_execs_exact_argv: Literal[True] = True
    submitted_command_string_or_shell_construction_allowed: Literal[False] = False
    inherited_environment_allowed: Literal[False] = False
    ambient_provider_route_allowed: Literal[False] = False
    ambient_credentials_allowed: Literal[False] = False
    provider_route_bound_outside_guest: Literal[True] = True
    image_and_receipt_bindings_are_external: Literal[True] = True
    linux_kvm_qualification_claimed: Literal[False] = False
    development_only: Literal[True] = True

    @field_validator('guest_executable_path', 'guest_config_path')
    @classmethod
    def validate_harness_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or '..' in path.parts
            or path.as_posix() != value
            or not value.startswith('/opt/vaxreplay/')
        ):
            raise ValueError('guest dispatch paths must be normalized beneath the read-only harness mount')
        return value

    @field_validator('guest_argv')
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            not item
            or '\x00' in item
            or len(item.encode('ascii', errors='ignore')) != len(item)
            or len(item) > 4096
            or re.fullmatch(_SAFE_DISPATCH_ARGUMENT_PATTERN, item) is None
            for item in value
        ):
            raise ValueError('guest dispatch argv must contain only bounded, shell-inert ASCII elements')
        return value

    @model_validator(mode='after')
    def validate_dispatch(self) -> Self:
        if self.guest_executable_path == self.guest_config_path:
            raise ValueError('guest executable and configuration paths must be distinct')
        if self.guest_argv != (
            self.guest_executable_path,
            GUEST_CONFIG_DIGEST_FLAG,
            self.guest_config_sha256,
        ):
            raise ValueError('guest dispatch argv must exactly bind the executable and configuration digest')

        native = self.family == HarnessFamily.VAXREPLAY_NATIVE
        development_adapter = self.family in {HarnessFamily.CODEX, HarnessFamily.CLAUDE_CODE}
        if not native and not development_adapter:
            raise ValueError('this harness family has no bootable guest-dispatch implementation')
        if native:
            expected = (
                HarnessRuntimeSupport.RUNTIME_INTEGRATED,
                GuestBootDispatchAdmission.RUNTIME_INTEGRATED_REQUIRES_EXTERNAL_QUALIFICATION,
                GuestBootConfigSchema.LANE_A_CLINICAL,
                NATIVE_GUEST_EXECUTABLE_PATH,
                NATIVE_GUEST_CONFIG_PATH,
            )
        else:
            expected = (
                HarnessRuntimeSupport.DEVELOPMENT_ADAPTER_INTEGRATED,
                GuestBootDispatchAdmission.DEVELOPMENT_PACKAGING_ONLY,
                GuestBootConfigSchema.HEADLESS_ADAPTER,
                HEADLESS_GUEST_EXECUTABLE_PATH,
                HEADLESS_GUEST_CONFIG_PATH,
            )
        observed = (
            self.runtime_support,
            self.admission,
            self.config_schema,
            self.guest_executable_path,
            self.guest_config_path,
        )
        if observed != expected:
            raise ValueError('guest dispatch implementation state, config schema, or fixed path is inconsistent')
        return self


def guest_boot_dispatch_manifest_sha256(manifest: GuestBootDispatchManifest) -> str:
    canonical = GuestBootDispatchManifest.model_validate_json(canonical_json_bytes(manifest))
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def make_native_guest_boot_dispatch_manifest(
    *,
    guest_executable_sha256: str,
    guest_config_sha256: str,
) -> GuestBootDispatchManifest:
    return GuestBootDispatchManifest(
        family=HarnessFamily.VAXREPLAY_NATIVE,
        runtime_support=HarnessRuntimeSupport.RUNTIME_INTEGRATED,
        admission=(GuestBootDispatchAdmission.RUNTIME_INTEGRATED_REQUIRES_EXTERNAL_QUALIFICATION),
        config_schema=GuestBootConfigSchema.LANE_A_CLINICAL,
        guest_executable_path=NATIVE_GUEST_EXECUTABLE_PATH,
        guest_executable_sha256=guest_executable_sha256,
        guest_config_path=NATIVE_GUEST_CONFIG_PATH,
        guest_config_sha256=guest_config_sha256,
        guest_argv=(
            NATIVE_GUEST_EXECUTABLE_PATH,
            GUEST_CONFIG_DIGEST_FLAG,
            guest_config_sha256,
        ),
    )


def load_pinned_guest_boot_dispatch_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> GuestBootDispatchManifest:
    payload = _read_regular_file_no_follow(path, maximum_bytes=_MAXIMUM_MANIFEST_BYTES)
    if not _valid_sha256(expected_sha256) or not hmac.compare_digest(
        hashlib.sha256(payload).hexdigest(), expected_sha256
    ):
        raise GuestBootDispatchError('guest boot-dispatch manifest differs from its external pin')
    try:
        manifest = GuestBootDispatchManifest.model_validate_json(payload)
    except (TypeError, ValueError) as error:
        raise GuestBootDispatchError('guest boot-dispatch manifest schema is invalid') from error
    if not hmac.compare_digest(payload, canonical_json_bytes(manifest)):
        raise GuestBootDispatchError('guest boot-dispatch manifest is not exact canonical JSON')
    return manifest


def load_and_validate_guest_boot_config(
    path: Path,
    *,
    dispatch: GuestBootDispatchManifest,
) -> bytes:
    payload = _read_regular_file_no_follow(path, maximum_bytes=_MAXIMUM_CONFIG_BYTES)
    validate_guest_boot_config_bytes(dispatch, payload)
    return payload


def validate_guest_boot_config_bytes(
    dispatch: GuestBootDispatchManifest,
    payload: bytes,
) -> None:
    """Require exact canonical config bytes for the manifest-declared implementation."""

    canonical_dispatch = GuestBootDispatchManifest.model_validate_json(canonical_json_bytes(dispatch))
    if (
        not payload
        or len(payload) > _MAXIMUM_CONFIG_BYTES
        or not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), canonical_dispatch.guest_config_sha256)
    ):
        raise GuestBootDispatchError('guest configuration differs from the dispatch manifest')
    try:
        if canonical_dispatch.config_schema == GuestBootConfigSchema.LANE_A_CLINICAL:
            from vaxreplay.agentic.clinical_guest_executable import LaneAClinicalGuestConfig

            config = LaneAClinicalGuestConfig.model_validate_json(payload)
        else:
            from vaxreplay.agentic.headless_guest_adapter import HeadlessGuestAdapterConfig

            config = HeadlessGuestAdapterConfig.model_validate_json(payload)
            if (
                config.family != canonical_dispatch.family
                or config.adapter_executable_path != canonical_dispatch.guest_executable_path
                or config.adapter_executable_sha256 != canonical_dispatch.guest_executable_sha256
                or config.adapter_implementation_checked_in is not True
                or config.linux_kvm_qualified is not False
                or config.development_only is not True
            ):
                raise GuestBootDispatchError('headless config differs from the development dispatch identity')
    except GuestBootDispatchError:
        raise
    except (TypeError, ValueError) as error:
        raise GuestBootDispatchError('guest configuration schema is invalid') from error
    if not hmac.compare_digest(payload, canonical_json_bytes(config)):
        raise GuestBootDispatchError('guest configuration is not exact canonical JSON')


def require_guest_boot_dispatch_binding(
    *,
    dispatch: GuestBootDispatchManifest,
    submitted_harness: SubmittedHarnessManifest,
) -> None:
    """Cross-bind pre-build launch facts to the externally finalized harness identity."""

    canonical = GuestBootDispatchManifest.model_validate_json(canonical_json_bytes(dispatch))
    submitted = SubmittedHarnessManifest.model_validate_json(canonical_json_bytes(submitted_harness))
    expected = (
        submitted.family,
        submitted.runtime_support,
        submitted.guest_executable_path,
        submitted.guest_executable_sha256,
        submitted.baked_config_sha256,
        submitted.guest_argv,
    )
    observed = (
        canonical.family,
        canonical.runtime_support,
        canonical.guest_executable_path,
        canonical.guest_executable_sha256,
        canonical.guest_config_sha256,
        canonical.guest_argv,
    )
    if observed != expected:
        raise GuestBootDispatchError('guest boot-dispatch manifest differs from the finalized submitted harness')


def _read_regular_file_no_follow(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    no_follow = getattr(os, 'O_NOFOLLOW', None)
    if not isinstance(no_follow, int):
        raise GuestBootDispatchError('no-follow file loading is unavailable')
    descriptor = -1
    try:
        descriptor = os.open(path, flags | no_follow)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise GuestBootDispatchError('guest dispatch artifact is not a bounded regular file')
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b''.join(chunks)
        after = os.fstat(descriptor)
    except GuestBootDispatchError:
        raise
    except OSError as error:
        raise GuestBootDispatchError('guest dispatch artifact cannot be read safely') from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        len(payload) != before.st_size
        or len(payload) > maximum_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise GuestBootDispatchError('guest dispatch artifact changed while it was read')
    return payload


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in '0123456789abcdef' for character in value)


__all__ = [
    'GUEST_BOOT_DISPATCH_ID',
    'GUEST_BOOT_DISPATCH_MANIFEST_SCHEMA_VERSION',
    'GUEST_CONFIG_DIGEST_FLAG',
    'HEADLESS_GUEST_CONFIG_PATH',
    'HEADLESS_GUEST_EXECUTABLE_PATH',
    'NATIVE_GUEST_CONFIG_PATH',
    'NATIVE_GUEST_EXECUTABLE_PATH',
    'GuestBootConfigSchema',
    'GuestBootDispatchAdmission',
    'GuestBootDispatchError',
    'GuestBootDispatchManifest',
    'guest_boot_dispatch_manifest_sha256',
    'load_and_validate_guest_boot_config',
    'load_pinned_guest_boot_dispatch_manifest',
    'make_native_guest_boot_dispatch_manifest',
    'require_guest_boot_dispatch_binding',
    'validate_guest_boot_config_bytes',
]
