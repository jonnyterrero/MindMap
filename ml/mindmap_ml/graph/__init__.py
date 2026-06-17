"""Verified mindmap generation from user text (3-stage pipeline).

Stage 1 (this package's ``ingest``): import & digest free text into immutable,
char-offset-addressable evidence spans. Stages 2 (generation) and 3 (external
verification) build on these spans. See ml/MINDMAP_PIPELINE_DESIGN.md.

The non-negotiable Stage-1 invariant is **offset integrity**: every span must
point back to the exact characters of the (canonicalized) raw text, so all
downstream provenance is trustworthy.
"""
