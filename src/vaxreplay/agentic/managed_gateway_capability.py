"""Restart-visible gateway capability cleanup bound to managed run ownership.

The managed ownership chain proves which capability belongs to which redeemed start.  The gateway
ledger proves the exact registered grant and route, or retains an exact pre-registration tombstone
when a process died between ``record_start_bound`` and gateway registration.  This adapter composes
those two authorities in the reaper's required order:

1. durably tombstone local gateway admission;
2. optionally clear an additional volatile/external capability namespace;
3. append the authenticated ownership ``capability_revoked`` successor.

The optional callback is defense in depth.  It does not replace the local tombstone and this module
does not claim to revoke a provider API key or cancel a request already dispatched remotely.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from datetime import UTC, datetime

from vaxreplay.agentic.managed_clinical_ownership import (
    DurableManagedClinicalOwnershipLedger,
    ManagedClinicalOwnershipError,
    authenticated_managed_clinical_ownership_sha256,
)
from vaxreplay.agentic.managed_clinical_startup import (
    ManagedClinicalCapability,
    managed_clinical_cleanup_key_id,
    managed_clinical_ownership_hmac,
)
from vaxreplay.agentic.provider_gateway import (
    AuthenticatedGatewayError,
    GatewayCapabilityBinding,
    GatewayCapabilityRevocationReason,
    SqliteGatewayLedger,
)


class RestartVisibleManagedGatewayCapabilityLedger:
    """Concrete startup-reaper capability inventory and durable local revoker."""

    def __init__(
        self,
        *,
        ownership: DurableManagedClinicalOwnershipLedger,
        ownership_key: bytes,
        gateway_ledger: SqliteGatewayLedger,
        expected_model_route_sha256: str,
        after_local_revocation: Callable[[str], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if managed_clinical_cleanup_key_id(ownership_key) != ownership.config.ownership_key_id:
            raise ValueError('managed gateway capability key differs from ownership key ID')
        if len(expected_model_route_sha256) != 64 or any(
            character not in '0123456789abcdef' for character in expected_model_route_sha256
        ):
            raise ValueError('managed gateway capability route SHA-256 is invalid')
        self.ownership = ownership
        self.gateway_ledger = gateway_ledger
        self.expected_model_route_sha256 = expected_model_route_sha256
        self._key = bytes(ownership_key)
        self._after_local_revocation = after_local_revocation
        self._clock = clock or (lambda: datetime.now(UTC))

    def inventory(self) -> tuple[ManagedClinicalCapability, ...]:
        values: list[ManagedClinicalCapability] = []
        owned_state_by_capability: dict[str, str] = {}
        for envelope in self.ownership.active():
            record = envelope.record
            if record.capability_id is None or record.capability_revoked:
                continue
            if record.start_redemption_sha256 is None:
                raise ManagedClinicalOwnershipError('active capability lacks its exact redeemed-start hash')
            if record.capability_id in owned_state_by_capability:
                raise ManagedClinicalOwnershipError('managed ownership contains a duplicate capability ID')
            owned_state_by_capability[record.capability_id] = record.state
            binding = self._optional_binding(record.capability_id)
            if binding is not None:
                self._validate_binding(
                    binding,
                    run_id=record.run_id,
                    start_redemption_sha256=record.start_redemption_sha256,
                )
            revocation = self.gateway_ledger.capability_revocation(record.capability_id)
            if revocation is not None and (
                revocation.run_id,
                revocation.attempt_reservation_sha256,
                revocation.model_route_sha256,
            ) != (
                record.run_id,
                record.start_redemption_sha256,
                self.expected_model_route_sha256,
            ):
                raise ManagedClinicalOwnershipError('durable capability tombstone differs from owned start and route')
            unsigned = ManagedClinicalCapability(
                capability_id=record.capability_id,
                run_id=record.run_id,
                registry_authority_id=record.registry_authority_id,
                reservation_sha256=record.reservation_sha256,
                launch_sha256=record.launch_sha256,
                start_redemption_sha256=record.start_redemption_sha256,
                worker_spec_sha256=record.worker_spec_sha256,
                ownership_record_sha256=(authenticated_managed_clinical_ownership_sha256(envelope)),
                ownership_authentication_hmac_sha256='0' * 64,
            )
            values.append(
                unsigned.model_copy(
                    update={
                        'ownership_authentication_hmac_sha256': (
                            managed_clinical_ownership_hmac(unsigned, key=self._key)
                        )
                    }
                )
            )
        unrevoked_bindings = self.gateway_ledger.unrevoked_capability_bindings()
        unrevoked = {binding.capability_id: binding for binding in unrevoked_bindings}
        if len(unrevoked) != len(unrevoked_bindings):
            raise ManagedClinicalOwnershipError('gateway ledger contains duplicate unrevoked capability IDs')
        extra = set(unrevoked) - set(owned_state_by_capability)
        if extra:
            raise ManagedClinicalOwnershipError('gateway ledger contains an unowned untombstoned capability')
        for capability in values:
            binding = unrevoked.get(capability.capability_id)
            if binding is not None:
                self._validate_binding(
                    binding,
                    run_id=capability.run_id,
                    start_redemption_sha256=capability.start_redemption_sha256,
                )
            elif (
                owned_state_by_capability[capability.capability_id] == 'running'
                and self.gateway_ledger.capability_revocation(capability.capability_id) is None
            ):
                raise ManagedClinicalOwnershipError(
                    'running ownership lacks its registered gateway capability or tombstone'
                )
        return tuple(sorted(values, key=lambda item: (item.run_id, item.capability_id)))

    def revoke(self, capability: ManagedClinicalCapability) -> None:
        if not hmac.compare_digest(
            capability.ownership_authentication_hmac_sha256,
            managed_clinical_ownership_hmac(capability, key=self._key),
        ):
            raise ManagedClinicalOwnershipError('capability revocation received unowned metadata')
        latest = self.ownership.latest(capability.run_id)
        record = latest.record
        if (
            record.capability_id != capability.capability_id
            or record.start_redemption_sha256 != capability.start_redemption_sha256
            or authenticated_managed_clinical_ownership_sha256(latest) != capability.ownership_record_sha256
        ):
            raise ManagedClinicalOwnershipError('capability revocation received stale metadata')

        revoked_at = self._now()
        binding = self._optional_binding(capability.capability_id)
        if binding is None:
            self.gateway_ledger.revoke_unregistered_capability(
                capability_id=capability.capability_id,
                expected_run_id=capability.run_id,
                # The gateway grant's unfortunately named field carries this exact redeemed-start
                # hash in the current runtime, not the outer cohort reservation hash.
                expected_attempt_reservation_sha256=capability.start_redemption_sha256,
                expected_model_route_sha256=self.expected_model_route_sha256,
                reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
                revoked_at=revoked_at,
            )
        else:
            self._validate_binding(
                binding,
                run_id=capability.run_id,
                start_redemption_sha256=capability.start_redemption_sha256,
            )
            self.gateway_ledger.revoke_capability(
                capability_id=capability.capability_id,
                expected_run_id=capability.run_id,
                expected_attempt_reservation_sha256=capability.start_redemption_sha256,
                expected_model_route_sha256=self.expected_model_route_sha256,
                reason=GatewayCapabilityRevocationReason.STARTUP_REAPER,
                revoked_at=revoked_at,
            )

        if self._after_local_revocation is not None:
            self._after_local_revocation(capability.capability_id)
        self.ownership.record_capability_revoked(
            run_id=capability.run_id,
            capability_id=capability.capability_id,
        )

    def _optional_binding(self, capability_id: str) -> GatewayCapabilityBinding | None:
        try:
            return self.gateway_ledger.capability_binding(capability_id)
        except AuthenticatedGatewayError:
            return None

    def _validate_binding(
        self,
        binding: GatewayCapabilityBinding,
        *,
        run_id: str,
        start_redemption_sha256: str,
    ) -> None:
        if (
            binding.run_id,
            binding.attempt_reservation_sha256,
            binding.model_route_sha256,
        ) != (
            run_id,
            start_redemption_sha256,
            self.expected_model_route_sha256,
        ):
            raise ManagedClinicalOwnershipError('gateway grant differs from the owned run, redeemed start, or route')

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ManagedClinicalOwnershipError('managed gateway capability clock must return an aware time')
        return value.astimezone(UTC)


__all__ = ['RestartVisibleManagedGatewayCapabilityLedger']
