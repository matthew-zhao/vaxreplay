# Fictional IEDB-shaped history

Every case record, reference, source record, antigen, and assay result in this directory is
project-authored and fictional. Project-authored entities use the visibly non-resolvable
`VAXREPLAY_FIXTURE_*` identifier namespaces; only correctly used public vocabulary identifiers,
such as OBI assay terms and the human NCBI Taxonomy identifier, are retained to exercise schema
compatibility. The files mimic the subset of IQ-API JSON fields consumed by the adapter and are
released as test data only. They contain no IEDB records, are not scientific evidence, and must not
be attributed to IEDB.

The project-authored fixture is licensed under `CC-BY-4.0`; retain the attribution specified in
the repository `DATA_LICENSES.md`.

The outcome snapshot intentionally contains:

- two new homogeneous T-cell assays used as private labels;
- a reported publication year older than the cutoff, proving publication year does not control
  availability; and
- a post-cutoff correction to a pre-existing assay, proving revisions are not counted as new assays.
