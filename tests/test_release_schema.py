from __future__ import annotations

import unittest

from pydantic import ValidationError

from vaxreplay.release_schema import (
    ChallengeAdmissionCommitment,
    ChallengeTemporalAdmissionBinding,
    PrivateFileBinding,
    PrivateReleaseManifest,
    ReleaseEpisodeBinding,
    ReleasePurpose,
)
from vaxreplay.temporal_schema import TemporalSourceTier


class OfficialReleaseSchemaTest(unittest.TestCase):
    def test_official_challenge_cannot_omit_complete_case_inventory(self) -> None:
        with self.assertRaisesRegex(ValidationError, 'complete sealed case inventory'):
            ChallengeAdmissionCommitment(
                release_id='official-1',
                purpose=ReleasePurpose.OFFICIAL_BENCHMARK,
                split_admission_sha256='a' * 64,
                split_inventory_complete=True,
                episodes=(
                    ChallengeTemporalAdmissionBinding(
                        episode_id='episode-1',
                        manifest_sha256='b' * 64,
                        source_tier=TemporalSourceTier.TIER_A,
                        temporal_admission_sha256='c' * 64,
                    ),
                ),
            )

    def test_official_private_release_cannot_omit_complete_case_inventory(self) -> None:
        with self.assertRaisesRegex(ValidationError, 'complete sealed case inventory'):
            PrivateReleaseManifest(
                release_id='official-1',
                purpose=ReleasePurpose.OFFICIAL_BENCHMARK,
                challenge_id='challenge-1',
                challenge_bundle_sha256='a' * 64,
                suite_manifest_sha256='b' * 64,
                admission_sha256='c' * 64,
                policy_sha256='d' * 64,
                receipt_key_id='e' * 64,
                split_admission_sha256='f' * 64,
                split_inventory_complete=True,
                episodes=(
                    ReleaseEpisodeBinding(
                        ordinal=0,
                        episode_id='episode-1',
                        private_path='episodes/000000',
                        manifest_sha256='1' * 64,
                        labels_sha256='2' * 64,
                        label_commitment_key_id='3' * 64,
                        temporal_admission_sha256='4' * 64,
                        source_tier=TemporalSourceTier.TIER_A,
                        source_audit_sha256='5' * 64,
                    ),
                ),
                files=(
                    PrivateFileBinding(
                        path='episodes/000000/manifest.json',
                        sha256='6' * 64,
                        byte_count=1,
                    ),
                ),
            )


if __name__ == '__main__':
    unittest.main()
