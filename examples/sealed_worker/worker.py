#!/usr/bin/env python3
"""Minimal protocol example that emits a uniform, citation-free VaxReplay submission."""

from __future__ import annotations

import json
import sys


def main() -> None:
    envelope = json.load(sys.stdin)
    prompt = envelope['messages'][1]['content']
    episode_text = prompt.split('EPISODE\n', maxsplit=1)[1].split('\n\nOUTPUT CONTRACT', maxsplit=1)[0]
    episode = json.loads(episode_text)
    candidate_ids = episode['candidate_ids']
    portfolio_size = episode['portfolio_size']
    forecasts = [
        {
            'candidate_id': candidate_id,
            'target_id': target['target_id'],
            'horizon_days': target['horizon_days'],
            'probability': 0.5,
        }
        for candidate_id in candidate_ids
        for target in episode['forecast_targets']
    ]
    assessments = [
        {
            'candidate_id': candidate_id,
            'dimension': dimension,
            'conclusion': 'insufficient',
            'citations': [],
        }
        for candidate_id in candidate_ids[:portfolio_size]
        for dimension in episode['required_dimensions']
    ]
    json.dump(
        {
            'schema_version': 'vaxreplay.v0.1',
            'episode_id': episode['episode_id'],
            'manifest_sha256': episode['manifest_sha256'],
            'ranking': candidate_ids,
            'forecasts': forecasts,
            'assessments': assessments,
        },
        sys.stdout,
        separators=(',', ':'),
        sort_keys=True,
    )


if __name__ == '__main__':
    main()
