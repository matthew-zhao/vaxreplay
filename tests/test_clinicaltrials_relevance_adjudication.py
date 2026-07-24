from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.clinicaltrials import relevance_adjudication
from vaxreplay.clinicaltrials.execution_inventory import bind_anchor_source, build_execution_inventory
from vaxreplay.clinicaltrials.execution_schema import (
    AactExecutionDecisionRow,
    DiseaseStratum,
    ExecutionCohortPolicy,
    NormalizedPhase,
    NormalizedStudyType,
    RegistryStatus,
    RegistryValueType,
)
from vaxreplay.clinicaltrials.relevance_adjudication import (
    ACTIVE_VACCINE_RELEVANCE_POLICY,
    RelevanceDisposition,
    RelevanceReason,
    RelevanceReviewError,
    RelevanceReviewInput,
    build_relevance_review_queue,
    finalize_relevance_adjudications,
    relevance_policy_sha256,
    verify_relevance_review_build,
    write_relevance_review_build,
)

ANCHOR = date(2020, 2, 1)
SNAPSHOT = 'aact-flatfiles-2020-02-01'
NCT_ID = 'NCT00000001'


def _zip(path: Path, *, brief_title: str = 'Safety of Candidate V1') -> str:
    members = {
        'brief_summaries.txt': 'id|nct_id|description\n1|NCT00000001|Candidate summary\n',
        'conditions.txt': 'id|nct_id|name|downcase_name\n1|NCT00000001|Influenza|influenza\n',
        'designs.txt': 'id|nct_id|primary_purpose\n1|NCT00000001|Prevention\n',
        'detailed_descriptions.txt': (
            'id|nct_id|description\n1|NCT00000001|"First line\nsecond line of candidate description"\n'
        ),
        'interventions.txt': (
            'id|nct_id|intervention_type|name|description\n'
            '1|NCT00000001|Biological|Candidate V1|Investigational antigen vaccine\n'
        ),
        'sponsors.txt': (
            'id|nct_id|agency_class|lead_or_collaborator|name\n1|NCT00000001|INDUSTRY|lead|Example Sponsor\n'
        ),
        'studies.txt': (
            'nct_id|acronym|brief_title|official_title\n'
            f'NCT00000001|V1|{brief_title}|Safety and Immunogenicity of Candidate V1\n'
        ),
    }
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(archive_sha256: str):
    row = AactExecutionDecisionRow(
        snapshot_id=SNAPSHOT,
        archive_date=ANCHOR,
        source_record_sha256='a' * 64,
        nct_id=NCT_ID,
        lineage_group_id='lin-example',
        disease_stratum=DiseaseStratum.NON_COVID_INFECTIOUS,
        study_first_posted_date=date(2020, 1, 1),
        study_type=NormalizedStudyType.INTERVENTIONAL,
        phase=NormalizedPhase.PHASE_1,
        human=True,
        prophylactic_intent=True,
        infectious_disease_vaccine=True,
        biological_intervention_count=1,
        overall_status=RegistryStatus.RECRUITING,
        results_section_present=False,
        enrollment=20,
        enrollment_type=RegistryValueType.ANTICIPATED,
        primary_completion_date=date(2021, 1, 1),
        primary_completion_date_type=RegistryValueType.ANTICIPATED,
    )
    binding = bind_anchor_source(
        anchor_date=ANCHOR,
        decision_snapshot_id=SNAPSHOT,
        decision_archive_manifest_sha256=archive_sha256,
        label_snapshot_id='aact-flatfiles-2024-02-01',
        label_archive_manifest_sha256='b' * 64,
        rows=(row,),
    )
    policy = ExecutionCohortPolicy(
        policy_id='test-policy',
        synthetic=True,
        selection_universe_rule_id='test-selection',
        selection_universe_rule_sha256='c' * 64,
        lineage_grouping_rule_id='test-lineage',
        lineage_grouping_rule_sha256='d' * 64,
        anchors=(binding,),
    )
    return build_execution_inventory(policy=policy, decision_rows=(row,))


def _queue(root: Path):
    archive = root / 'decision.zip'
    archive_sha256 = _zip(archive)
    inventory = _inventory(archive_sha256)
    inventory_sha256 = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    return (
        build_relevance_review_queue(
            inventory=inventory,
            merged_inventory_sha256=inventory_sha256,
            decision_archives={ANCHOR: archive},
        ),
        archive,
    )


def _include_review(queue) -> RelevanceReviewInput:
    record = queue.records[0]
    return RelevanceReviewInput(
        nct_id=record.nct_id,
        anchor_date=record.anchor_date,
        evidence_sha256=record.evidence_sha256,
        disposition=RelevanceDisposition.INCLUDE,
        reason_codes=(RelevanceReason.INCLUDE_ACTIVE_PROPHYLACTIC_VACCINE_CANDIDATE,),
        rationale='Decision-time text identifies an investigational antigen vaccine candidate.',
    )


class RelevanceAdjudicationTests(unittest.TestCase):
    def test_queue_binds_only_exact_decision_text_and_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            queue, _ = _queue(Path(temporary))

        self.assertEqual(queue.record_count, 1)
        self.assertEqual(queue.policy_sha256, relevance_policy_sha256())
        self.assertEqual(
            relevance_policy_sha256(), hashlib.sha256(canonical_json_bytes(ACTIVE_VACCINE_RELEVANCE_POLICY)).hexdigest()
        )
        self.assertTrue(queue.decision_archives_only)
        self.assertFalse(queue.later_archive_opened)
        self.assertFalse(queue.execution_labels_read)
        record = queue.records[0]
        self.assertEqual(record.brief_title, 'Safety of Candidate V1')
        self.assertIn('second line', record.detailed_description)
        self.assertEqual(record.interventions[0].name, 'Candidate V1')
        self.assertEqual(record.sponsors[0].name, 'Example Sponsor')
        self.assertGreaterEqual(len(record.source_rows), 7)

    def test_archive_must_match_inventory_decision_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, archive = _queue(root)
            archive.write_bytes(archive.read_bytes() + b'tamper')
            inventory = _inventory(queue.source_archives[0].archive_sha256)
            with self.assertRaisesRegex(RelevanceReviewError, 'SHA-256'):
                build_relevance_review_queue(
                    inventory=inventory,
                    merged_inventory_sha256=hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(),
                    decision_archives={ANCHOR: archive},
                )

    def test_inventory_hash_cannot_be_caller_asserted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'decision.zip'
            inventory = _inventory(_zip(archive))
            with self.assertRaisesRegex(RelevanceReviewError, 'canonical decision inventory'):
                build_relevance_review_queue(
                    inventory=inventory,
                    merged_inventory_sha256='a' * 64,
                    decision_archives={ANCHOR: archive},
                )

    def test_finalize_requires_exact_coverage_and_evidence_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            queue, _ = _queue(Path(temporary))
        with self.assertRaisesRegex(RelevanceReviewError, 'cover every queue'):
            finalize_relevance_adjudications(queue=queue, reviews=())
        review = _include_review(queue).model_copy(update={'evidence_sha256': 'e' * 64})
        with self.assertRaisesRegex(RelevanceReviewError, 'evidence hash'):
            finalize_relevance_adjudications(queue=queue, reviews=(review,))

    def test_reason_must_match_disposition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            queue, _ = _queue(Path(temporary))
        record = queue.records[0]
        with self.assertRaises(ValidationError):
            RelevanceReviewInput(
                nct_id=record.nct_id,
                anchor_date=record.anchor_date,
                evidence_sha256=record.evidence_sha256,
                disposition=RelevanceDisposition.INCLUDE,
                reason_codes=(RelevanceReason.EXCLUDE_PASSIVE_ANTIBODY,),
                rationale='Invalid combination.',
            )

    def test_content_addressed_build_verifies_and_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, archive = _queue(root)
            inventory = _inventory(queue.source_archives[0].archive_sha256)
            inventory_path = root / 'inventory.json'
            inventory_path.write_bytes(canonical_json_bytes(inventory))
            output = root / 'review'
            receipt = write_relevance_review_build(
                queue=queue,
                reviews=(_include_review(queue),),
                output_root=output,
            )
            self.assertEqual(receipt.include_count, 1)
            receipt_sha256 = hashlib.sha256((output / 'REVIEW-RECEIPT.json').read_bytes()).hexdigest()
            self.assertEqual(
                verify_relevance_review_build(
                    output,
                    expected_receipt_sha256=receipt_sha256,
                    trusted_inventory_path=inventory_path,
                    trusted_decision_archives={ANCHOR: archive},
                ),
                receipt,
            )
            queue_path = output / 'organizer' / 'relevance-review-queue.json'
            queue_path.write_bytes(queue_path.read_bytes() + b' ')
            with self.assertRaisesRegex(RelevanceReviewError, 'does not match receipt'):
                verify_relevance_review_build(
                    output,
                    expected_receipt_sha256=receipt_sha256,
                    trusted_inventory_path=inventory_path,
                    trusted_decision_archives={ANCHOR: archive},
                )

    def test_self_consistent_forged_sources_do_not_verify_against_trusted_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trusted_queue, trusted_archive = _queue(root)
            trusted_inventory = _inventory(trusted_queue.source_archives[0].archive_sha256)
            trusted_inventory_path = root / 'trusted-inventory.json'
            trusted_inventory_path.write_bytes(canonical_json_bytes(trusted_inventory))

            forged_archive = root / 'forged-decision.zip'
            forged_inventory = _inventory(_zip(forged_archive, brief_title='Forged Candidate Title'))
            forged_queue = build_relevance_review_queue(
                inventory=forged_inventory,
                merged_inventory_sha256=hashlib.sha256(canonical_json_bytes(forged_inventory)).hexdigest(),
                decision_archives={ANCHOR: forged_archive},
            )
            forged_output = root / 'forged-review'
            write_relevance_review_build(
                queue=forged_queue,
                reviews=(_include_review(forged_queue),),
                output_root=forged_output,
            )
            trusted_output = root / 'trusted-review'
            write_relevance_review_build(
                queue=trusted_queue,
                reviews=(_include_review(trusted_queue),),
                output_root=trusted_output,
            )
            trusted_receipt_sha256 = hashlib.sha256((trusted_output / 'REVIEW-RECEIPT.json').read_bytes()).hexdigest()
            with self.assertRaisesRegex(RelevanceReviewError, 'externally pinned digest'):
                verify_relevance_review_build(
                    forged_output,
                    expected_receipt_sha256=trusted_receipt_sha256,
                    trusted_inventory_path=trusted_inventory_path,
                    trusted_decision_archives={ANCHOR: trusted_archive},
                )
            with self.assertRaisesRegex(RelevanceReviewError, 'does not reconstruct from the trusted'):
                verify_relevance_review_build(
                    forged_output,
                    expected_receipt_sha256=hashlib.sha256(
                        (forged_output / 'REVIEW-RECEIPT.json').read_bytes()
                    ).hexdigest(),
                    trusted_inventory_path=trusted_inventory_path,
                    trusted_decision_archives={ANCHOR: trusted_archive},
                )

    def test_verifier_parses_each_hash_checked_artifact_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, archive = _queue(root)
            inventory = _inventory(queue.source_archives[0].archive_sha256)
            inventory_path = root / 'inventory.json'
            inventory_path.write_bytes(canonical_json_bytes(inventory))
            output = root / 'review'
            write_relevance_review_build(
                queue=queue,
                reviews=(_include_review(queue),),
                output_root=output,
            )
            receipt_sha256 = hashlib.sha256((output / 'REVIEW-RECEIPT.json').read_bytes()).hexdigest()
            adjudication_path = output / 'organizer' / 'relevance-adjudications.json'
            path_type = type(adjudication_path)
            original_read_bytes = path_type.read_bytes
            adjudication_reads = 0

            def changing_read_bytes(path: Path) -> bytes:
                nonlocal adjudication_reads
                if path.resolve() == adjudication_path.resolve():
                    adjudication_reads += 1
                    if adjudication_reads > 1:
                        return b'{"forged":"second read"}'
                return original_read_bytes(path)

            with patch.object(path_type, 'read_bytes', changing_read_bytes):
                verify_relevance_review_build(
                    output,
                    expected_receipt_sha256=receipt_sha256,
                    trusted_inventory_path=inventory_path,
                    trusted_decision_archives={ANCHOR: archive},
                )
            self.assertEqual(adjudication_reads, 1)

    def test_archive_path_replacement_during_scan_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, archive = _queue(root)
            inventory = _inventory(queue.source_archives[0].archive_sha256)
            inventory_path = root / 'inventory.json'
            inventory_path.write_bytes(canonical_json_bytes(inventory))
            output = root / 'review'
            write_relevance_review_build(
                queue=queue,
                reviews=(_include_review(queue),),
                output_root=output,
            )
            receipt_sha256 = hashlib.sha256((output / 'REVIEW-RECEIPT.json').read_bytes()).hexdigest()
            replacement = root / 'replacement.zip'
            _zip(replacement, brief_title='Replacement Candidate')
            original_hash = relevance_adjudication._hash_seekable_file
            replaced = False

            def replacing_hash(source):
                nonlocal replaced
                result = original_hash(source)
                if not replaced:
                    replaced = True
                    os.replace(replacement, archive)
                return result

            with (
                patch.object(relevance_adjudication, '_hash_seekable_file', replacing_hash),
                self.assertRaisesRegex(RelevanceReviewError, 'changed during verification'),
            ):
                verify_relevance_review_build(
                    output,
                    expected_receipt_sha256=receipt_sha256,
                    trusted_inventory_path=inventory_path,
                    trusted_decision_archives={ANCHOR: archive},
                )


if __name__ == '__main__':
    unittest.main()
