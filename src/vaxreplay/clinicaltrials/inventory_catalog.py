"""Deterministic parser and cross-verifier for official AACT archive listings."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, time, timezone
from html.parser import HTMLParser
from typing import Final

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials.inventory_schema import (
    AACT_CATALOG_PARSER_ID,
    AactArchiveAcquisitionItem,
    AactArchiveAcquisitionPlan,
    AactArchiveAcquisitionRole,
    AactOfficialArchiveCatalog,
    AactOfficialArchiveEntry,
    AactOfficialCatalogPage,
    aact_inventory_model_sha256,
    aact_inventory_records_sha256,
)

_OFFICIAL_ORIGIN: Final = 'https://aact.ctti-clinicaltrials.org'
_FILE_NAME_RE: Final = re.compile(r'^(?P<date>[0-9]{8})_(?:pipe-delimited-export|export|export_ctgov)\.zip$')


class AactCatalogIntegrityError(ValueError):
    """A frozen official archive listing or normalized catalog failed closed."""


@dataclass(frozen=True)
class FrozenOfficialArchiveListing:
    """Exact HTML bytes plus independently supplied retrieval facts."""

    year: int
    source_url: str
    retrieved_at: datetime
    payload: bytes


@dataclass(frozen=True)
class _RawArchiveRow:
    date_text: str
    file_name: str
    displayed_size: str
    download_path: str


class _ArchiveListingHtmlParser(HTMLParser):
    """Extract only rows following the official ``Monthly Archives`` heading."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.section: str | None = None
        self.heading_parts: list[str] | None = None
        self.active_row: dict[str, str] | None = None
        self.row_depth = 0
        self.active_cell: str | None = None
        self.active_cell_depth: int | None = None
        self.rows: list[_RawArchiveRow] = []
        self.saw_monthly_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = frozenset((attributes.get('class') or '').split())
        if tag == 'h2' and 'snapshots-section-title' in classes:
            self.heading_parts = []
            return

        if self.active_row is not None:
            if tag == 'div':
                self.row_depth += 1
                label = attributes.get('data-label')
                if label in {'Date:', 'File:', 'Size:'}:
                    if self.active_cell is not None:
                        raise AactCatalogIntegrityError('nested labeled cells in an archive listing row')
                    self.active_cell = label[:-1].casefold()
                    self.active_cell_depth = self.row_depth
            if tag == 'a' and attributes.get('href'):
                if 'download_path' in self.active_row:
                    raise AactCatalogIntegrityError('archive listing row contains multiple download links')
                self.active_row['download_path'] = attributes['href'] or ''
            return

        if tag == 'div' and 'snapshots-grid-row' in classes and 'snapshots-grid-header' not in classes:
            if self.section == 'monthly':
                self.active_row = {}
                self.row_depth = 1

    def handle_data(self, data: str) -> None:
        if self.heading_parts is not None:
            self.heading_parts.append(data)
        if self.active_row is not None and self.active_cell is not None:
            self.active_row[self.active_cell] = self.active_row.get(self.active_cell, '') + data

    def handle_endtag(self, tag: str) -> None:
        if tag == 'h2' and self.heading_parts is not None:
            heading = ' '.join(''.join(self.heading_parts).split())
            self.heading_parts = None
            if heading == 'Monthly Archives':
                self.section = 'monthly'
                self.saw_monthly_heading = True
            elif heading == 'Recent Daily Snapshots':
                self.section = 'recent'
            else:
                self.section = None
            return

        if self.active_row is None or tag != 'div':
            return
        if self.active_cell_depth == self.row_depth:
            self.active_cell = None
            self.active_cell_depth = None
        self.row_depth -= 1
        if self.row_depth:
            return

        required = {'date', 'file', 'size', 'download_path'}
        if set(self.active_row) != required:
            raise AactCatalogIntegrityError(
                f'archive listing row has fields {sorted(self.active_row)}, expected {sorted(required)}'
            )
        self.rows.append(
            _RawArchiveRow(
                date_text=' '.join(self.active_row['date'].split()),
                file_name=' '.join(self.active_row['file'].split()),
                displayed_size=' '.join(self.active_row['size'].split()),
                download_path=self.active_row['download_path'].strip(),
            )
        )
        self.active_row = None

    def close(self) -> None:
        super().close()
        if self.heading_parts is not None or self.active_row is not None:
            raise AactCatalogIntegrityError('archive listing HTML ended inside a heading or row')
        if not self.saw_monthly_heading:
            raise AactCatalogIntegrityError('archive listing does not contain the Monthly Archives section')
        if not self.rows:
            raise AactCatalogIntegrityError('archive listing contains no permanent monthly archive rows')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_listing(
    listing: FrozenOfficialArchiveListing,
) -> tuple[AactOfficialCatalogPage, tuple[AactOfficialArchiveEntry, ...]]:
    try:
        html = listing.payload.decode('utf-8')
    except UnicodeDecodeError as error:
        raise AactCatalogIntegrityError('official archive listing must be UTF-8 HTML') from error
    if not html or '\x00' in html:
        raise AactCatalogIntegrityError('official archive listing must be nonempty and NUL-free')

    payload_sha256 = _sha256(listing.payload)
    try:
        page = AactOfficialCatalogPage(
            year=listing.year,
            source_url=listing.source_url,
            retrieved_at=listing.retrieved_at,
            payload_sha256=payload_sha256,
            payload_bytes=len(listing.payload),
        )
    except ValueError as error:
        raise AactCatalogIntegrityError(f'invalid archive-listing receipt: {error}') from error

    parser = _ArchiveListingHtmlParser()
    try:
        parser.feed(html)
        parser.close()
    except (AactCatalogIntegrityError, ValueError) as error:
        raise AactCatalogIntegrityError(f'invalid official archive listing for {listing.year}: {error}') from error

    entries: list[AactOfficialArchiveEntry] = []
    for row in parser.rows:
        try:
            archive_date = datetime.strptime(row.date_text, '%m-%d-%Y').date()
        except ValueError as error:
            raise AactCatalogIntegrityError(f'invalid archive date {row.date_text!r}') from error
        match = _FILE_NAME_RE.fullmatch(row.file_name)
        if match is None:
            raise AactCatalogIntegrityError(f'unrecognized permanent archive file name: {row.file_name!r}')
        file_name_date = datetime.strptime(match.group('date'), '%Y%m%d').date()
        if archive_date.year != listing.year:
            raise AactCatalogIntegrityError('monthly archive row is outside its selected listing year')
        if file_name_date.year != listing.year:
            raise AactCatalogIntegrityError('monthly archive file-name date is outside its selected listing year')
        expected_path = f'/static/exported_files/daily/{archive_date.isoformat()}?source=web'
        if row.download_path != expected_path:
            raise AactCatalogIntegrityError('monthly archive download path is not the exact official dated route')
        try:
            entries.append(
                AactOfficialArchiveEntry(
                    snapshot_id=f'aact-flatfiles-{archive_date.isoformat()}',
                    archive_date=archive_date,
                    source_cutoff_at=datetime.combine(archive_date, time.max, tzinfo=timezone.utc),
                    listing_year=listing.year,
                    file_name=row.file_name,
                    file_name_date=file_name_date,
                    file_name_date_matches_archive_date=file_name_date == archive_date,
                    displayed_size=row.displayed_size,
                    download_path=row.download_path,
                    source_url=f'{_OFFICIAL_ORIGIN}{row.download_path}',
                    listing_page_sha256=payload_sha256,
                )
            )
        except ValueError as error:
            raise AactCatalogIntegrityError(f'invalid normalized archive entry: {error}') from error

    keys = [(entry.archive_date, entry.snapshot_id) for entry in entries]
    if len(keys) != len(set(keys)):
        raise AactCatalogIntegrityError('official listing contains a duplicate monthly archive')
    return page, tuple(sorted(entries, key=lambda entry: (entry.archive_date, entry.snapshot_id)))


def build_official_archive_catalog(
    *,
    catalog_id: str,
    generated_at: datetime,
    parser_implementation_sha256: str,
    listings: tuple[FrozenOfficialArchiveListing, ...],
) -> AactOfficialArchiveCatalog:
    """Parse frozen official listing bytes into one deterministic canonical catalog."""

    if not listings:
        raise AactCatalogIntegrityError('at least one frozen official listing is required')
    years = tuple(listing.year for listing in listings)
    if len(years) != len(set(years)):
        raise AactCatalogIntegrityError('frozen official listings must have unique years')

    parsed = [_parse_listing(listing) for listing in listings]
    pages = tuple(sorted((page for page, _ in parsed), key=lambda page: page.year))
    entries = tuple(
        sorted(
            (entry for _, page_entries in parsed for entry in page_entries),
            key=lambda entry: (entry.archive_date, entry.snapshot_id),
        )
    )
    if len({entry.snapshot_id for entry in entries}) != len(entries):
        raise AactCatalogIntegrityError('archive catalog contains a duplicate snapshot across annual pages')
    try:
        return AactOfficialArchiveCatalog(
            catalog_id=catalog_id,
            generated_at=generated_at,
            parser_id=AACT_CATALOG_PARSER_ID,
            parser_implementation_sha256=parser_implementation_sha256,
            pages=pages,
            source_pages_sha256=aact_inventory_records_sha256(pages),
            entries=entries,
            entries_sha256=aact_inventory_records_sha256(entries),
        )
    except ValueError as error:
        raise AactCatalogIntegrityError(f'invalid normalized archive catalog: {error}') from error


def verify_official_archive_catalog(
    catalog: AactOfficialArchiveCatalog,
    listings: tuple[FrozenOfficialArchiveListing, ...],
) -> None:
    """Reparse exact listing bytes and require canonical equality with a supplied catalog."""

    rebuilt = build_official_archive_catalog(
        catalog_id=catalog.catalog_id,
        generated_at=catalog.generated_at,
        parser_implementation_sha256=catalog.parser_implementation_sha256,
        listings=listings,
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(catalog):
        raise AactCatalogIntegrityError('archive catalog does not match the supplied frozen official listings')


def build_archive_acquisition_plan(
    *,
    plan_id: str,
    created_at: datetime,
    catalog: AactOfficialArchiveCatalog,
    screening_policy_sha256: str,
    requested_roles: dict[str, tuple[AactArchiveAcquisitionRole, ...]],
) -> AactArchiveAcquisitionPlan:
    """Bind requested archive roles to exact entries from one frozen official catalog."""

    if not requested_roles:
        raise AactCatalogIntegrityError('an acquisition plan must request at least one archive')
    entry_by_snapshot = {entry.snapshot_id: entry for entry in catalog.entries}
    unknown = set(requested_roles) - set(entry_by_snapshot)
    if unknown:
        raise AactCatalogIntegrityError(f'acquisition plan references unknown snapshots: {sorted(unknown)}')

    items: list[AactArchiveAcquisitionItem] = []
    for snapshot_id, roles in requested_roles.items():
        entry = entry_by_snapshot[snapshot_id]
        if not roles or len(roles) != len(set(roles)):
            raise AactCatalogIntegrityError('each acquisition item requires unique nonempty roles')
        ordered_roles = tuple(sorted(roles, key=lambda role: role.value))
        try:
            items.append(
                AactArchiveAcquisitionItem(
                    snapshot_id=entry.snapshot_id,
                    archive_date=entry.archive_date,
                    catalog_entry_sha256=aact_inventory_model_sha256(entry),
                    roles=ordered_roles,
                    target_relative_path=f'archives/{entry.file_name}',
                )
            )
        except ValueError as error:
            raise AactCatalogIntegrityError(f'invalid archive acquisition item: {error}') from error
    ordered_items = tuple(sorted(items, key=lambda item: (item.archive_date, item.snapshot_id)))
    try:
        plan = AactArchiveAcquisitionPlan(
            plan_id=plan_id,
            created_at=created_at,
            catalog_sha256=aact_inventory_model_sha256(catalog),
            catalog_entries_sha256=catalog.entries_sha256,
            screening_policy_sha256=screening_policy_sha256,
            items=ordered_items,
            items_sha256=aact_inventory_records_sha256(ordered_items),
        )
    except ValueError as error:
        raise AactCatalogIntegrityError(f'invalid archive acquisition plan: {error}') from error
    verify_archive_acquisition_plan(plan, catalog)
    return plan


def verify_archive_acquisition_plan(
    plan: AactArchiveAcquisitionPlan,
    catalog: AactOfficialArchiveCatalog,
) -> None:
    """Recheck every acquisition item against its exact catalog entry."""

    if plan.catalog_sha256 != aact_inventory_model_sha256(catalog):
        raise AactCatalogIntegrityError('acquisition plan binds a different archive catalog')
    if plan.created_at < catalog.generated_at:
        raise AactCatalogIntegrityError('acquisition plan cannot predate its archive catalog')
    if plan.catalog_entries_sha256 != catalog.entries_sha256:
        raise AactCatalogIntegrityError('acquisition plan binds a different archive-entry inventory')
    entry_by_snapshot = {entry.snapshot_id: entry for entry in catalog.entries}
    for item in plan.items:
        entry = entry_by_snapshot.get(item.snapshot_id)
        if entry is None:
            raise AactCatalogIntegrityError(f'acquisition item references unknown snapshot {item.snapshot_id}')
        if item.archive_date != entry.archive_date:
            raise AactCatalogIntegrityError('acquisition item archive date differs from its catalog entry')
        if item.catalog_entry_sha256 != aact_inventory_model_sha256(entry):
            raise AactCatalogIntegrityError('acquisition item hash differs from its catalog entry')
        if item.target_relative_path != f'archives/{entry.file_name}':
            raise AactCatalogIntegrityError('acquisition target path differs from the catalog file name')
    try:
        AactArchiveAcquisitionPlan.model_validate(plan.model_dump(mode='python'))
    except ValueError as error:
        raise AactCatalogIntegrityError(f'invalid acquisition plan structure: {error}') from error
