"""Offline organizer CLI for freezing AACT archive catalogs and acquisition plans.

The CLI performs no network access.  Organizers first retain annual official listing HTML with their
normal capture machinery, then supply those exact files through a small JSON build specification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_schema import StrictModel
from vaxreplay.clinicaltrials.inventory_catalog import (
    FrozenOfficialArchiveListing,
    build_archive_acquisition_plan,
    build_official_archive_catalog,
    verify_archive_acquisition_plan,
    verify_official_archive_catalog,
)
from vaxreplay.clinicaltrials.inventory_schema import (
    AactArchiveAcquisitionPlan,
    AactArchiveAcquisitionRole,
    AactOfficialArchiveCatalog,
    aact_inventory_model_sha256,
)


class AactInventoryCliError(ValueError):
    """An organizer CLI input is malformed, noncanonical, or unsafe to write."""


def _load_json_object(path: Path) -> tuple[Path, dict[str, Any]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AactInventoryCliError(f'input cannot be a symbolic link: {expanded}')
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise AactInventoryCliError(f'input must be a regular file: {resolved}')
    try:
        value = json.loads(resolved.read_bytes())
    except (OSError, ValueError) as error:
        raise AactInventoryCliError(f'cannot parse JSON input {resolved}: {error}') from error
    if not isinstance(value, dict):
        raise AactInventoryCliError(f'JSON input must be an object: {resolved}')
    return resolved, value


def _require_keys(value: dict[str, Any], expected: frozenset[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise AactInventoryCliError(f'{context} keys differ; missing={missing}, extra={extra}')


def _timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise AactInventoryCliError(f'{field_name} must be an ISO-8601 string')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise AactInventoryCliError(f'{field_name} is not a valid ISO-8601 timestamp') from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AactInventoryCliError(f'{field_name} must include a UTC offset')
    return parsed


def _read_regular_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise AactInventoryCliError(f'listing input cannot be a symbolic link: {path}')
    resolved = path.resolve()
    if not resolved.is_file():
        raise AactInventoryCliError(f'listing input must be a regular file: {resolved}')
    return resolved.read_bytes()


def _canonical_model_from_path[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AactInventoryCliError(f'model input cannot be a symbolic link: {expanded}')
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise AactInventoryCliError(f'model input must be a regular file: {resolved}')
    try:
        payload = resolved.read_bytes()
        parsed = model.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise AactInventoryCliError(f'cannot parse {model.__name__} from {resolved}: {error}') from error
    if payload != canonical_json_bytes(parsed) + b'\n':
        raise AactInventoryCliError(f'{resolved} is not exact canonical JSON with one trailing LF')
    return parsed


def _write_new_model(path: Path, value: BaseModel) -> None:
    target = path.expanduser().resolve()
    if not target.parent.is_dir():
        raise AactInventoryCliError(f'output parent must already exist: {target.parent}')
    payload = canonical_json_bytes(value) + b'\n'
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        raise AactInventoryCliError(f'refusing to replace or create output {target}: {error}') from error
    with os.fdopen(descriptor, 'wb') as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _catalog_from_spec(spec_path: Path) -> tuple[AactOfficialArchiveCatalog, tuple[FrozenOfficialArchiveListing, ...]]:
    resolved_spec, spec = _load_json_object(spec_path)
    _require_keys(
        spec,
        frozenset({'catalog_id', 'generated_at', 'parser_implementation_sha256', 'listings'}),
        'catalog spec',
    )
    listing_specs = spec['listings']
    if not isinstance(listing_specs, list) or not listing_specs:
        raise AactInventoryCliError('catalog spec listings must be a nonempty array')
    listings: list[FrozenOfficialArchiveListing] = []
    for index, raw_value in enumerate(listing_specs):
        if not isinstance(raw_value, dict):
            raise AactInventoryCliError(f'catalog listing {index} must be an object')
        raw = cast(dict[str, Any], raw_value)
        _require_keys(raw, frozenset({'year', 'source_url', 'retrieved_at', 'path'}), f'catalog listing {index}')
        if not isinstance(raw['year'], int) or isinstance(raw['year'], bool):
            raise AactInventoryCliError(f'catalog listing {index} year must be an integer')
        if not isinstance(raw['source_url'], str) or not isinstance(raw['path'], str):
            raise AactInventoryCliError(f'catalog listing {index} URL and path must be strings')
        listing_path = resolved_spec.parent / raw['path']
        listings.append(
            FrozenOfficialArchiveListing(
                year=raw['year'],
                source_url=raw['source_url'],
                retrieved_at=_timestamp(raw['retrieved_at'], f'catalog listing {index} retrieved_at'),
                payload=_read_regular_bytes(listing_path),
            )
        )
    if not isinstance(spec['catalog_id'], str) or not isinstance(spec['parser_implementation_sha256'], str):
        raise AactInventoryCliError('catalog ID and parser implementation hash must be strings')
    catalog = build_official_archive_catalog(
        catalog_id=spec['catalog_id'],
        generated_at=_timestamp(spec['generated_at'], 'generated_at'),
        parser_implementation_sha256=spec['parser_implementation_sha256'],
        listings=tuple(listings),
    )
    return catalog, tuple(listings)


def _plan_from_spec(
    spec_path: Path,
    catalog: AactOfficialArchiveCatalog,
) -> AactArchiveAcquisitionPlan:
    _, spec = _load_json_object(spec_path)
    _require_keys(
        spec,
        frozenset({'plan_id', 'created_at', 'screening_policy_sha256', 'requested_archives'}),
        'acquisition plan spec',
    )
    requests = spec['requested_archives']
    if not isinstance(requests, list) or not requests:
        raise AactInventoryCliError('requested_archives must be a nonempty array')
    requested_roles: dict[str, tuple[AactArchiveAcquisitionRole, ...]] = {}
    for index, raw_value in enumerate(requests):
        if not isinstance(raw_value, dict):
            raise AactInventoryCliError(f'acquisition request {index} must be an object')
        raw = cast(dict[str, Any], raw_value)
        _require_keys(raw, frozenset({'snapshot_id', 'roles'}), f'acquisition request {index}')
        if not isinstance(raw['snapshot_id'], str) or not isinstance(raw['roles'], list):
            raise AactInventoryCliError(f'acquisition request {index} has invalid snapshot ID or roles')
        if raw['snapshot_id'] in requested_roles:
            raise AactInventoryCliError(f'duplicate acquisition request for {raw["snapshot_id"]}')
        try:
            roles = tuple(AactArchiveAcquisitionRole(role) for role in raw['roles'])
        except (TypeError, ValueError) as error:
            raise AactInventoryCliError(f'acquisition request {index} has an invalid role') from error
        requested_roles[raw['snapshot_id']] = roles
    if not isinstance(spec['plan_id'], str) or not isinstance(spec['screening_policy_sha256'], str):
        raise AactInventoryCliError('plan ID and screening policy hash must be strings')
    return build_archive_acquisition_plan(
        plan_id=spec['plan_id'],
        created_at=_timestamp(spec['created_at'], 'created_at'),
        catalog=catalog,
        screening_policy_sha256=spec['screening_policy_sha256'],
        requested_roles=requested_roles,
    )


def _emit_status(kind: str, value: StrictModel) -> None:
    sys.stdout.buffer.write(
        canonical_json_bytes({'kind': kind, 'sha256': aact_inventory_model_sha256(value), 'valid': True}) + b'\n'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    build_catalog = commands.add_parser('build-catalog', help='build a catalog from frozen annual HTML')
    build_catalog.add_argument('--spec', type=Path, required=True)
    build_catalog.add_argument('--output', type=Path, required=True)

    verify_catalog = commands.add_parser('verify-catalog', help='reparse HTML and verify a catalog')
    verify_catalog.add_argument('--spec', type=Path, required=True)
    verify_catalog.add_argument('--catalog', type=Path, required=True)

    build_plan = commands.add_parser('build-plan', help='build an exact archive acquisition plan')
    build_plan.add_argument('--catalog', type=Path, required=True)
    build_plan.add_argument('--spec', type=Path, required=True)
    build_plan.add_argument('--output', type=Path, required=True)

    verify_plan = commands.add_parser('verify-plan', help='verify an archive plan against its catalog')
    verify_plan.add_argument('--catalog', type=Path, required=True)
    verify_plan.add_argument('--plan', type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == 'build-catalog':
            catalog, _ = _catalog_from_spec(args.spec)
            _write_new_model(args.output, catalog)
            _emit_status('aact_official_archive_catalog', catalog)
        elif args.command == 'verify-catalog':
            expected, listings = _catalog_from_spec(args.spec)
            catalog = _canonical_model_from_path(args.catalog, AactOfficialArchiveCatalog)
            verify_official_archive_catalog(catalog, listings)
            if canonical_json_bytes(catalog) != canonical_json_bytes(expected):
                raise AactInventoryCliError('catalog differs from the supplied build specification')
            _emit_status('aact_official_archive_catalog', catalog)
        elif args.command == 'build-plan':
            catalog = _canonical_model_from_path(args.catalog, AactOfficialArchiveCatalog)
            plan = _plan_from_spec(args.spec, catalog)
            _write_new_model(args.output, plan)
            _emit_status('aact_archive_acquisition_plan', plan)
        else:
            catalog = _canonical_model_from_path(args.catalog, AactOfficialArchiveCatalog)
            plan = _canonical_model_from_path(args.plan, AactArchiveAcquisitionPlan)
            verify_archive_acquisition_plan(plan, catalog)
            _emit_status('aact_archive_acquisition_plan', plan)
    except (AactInventoryCliError, ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
