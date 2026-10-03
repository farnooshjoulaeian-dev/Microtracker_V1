# Project collaboration instructions

The user is developing scientific software and wants teaching, scientific review,
and incremental assistance. Explain relevant concepts and tradeoffs before coding.
Keep the user responsible for scientific and architectural choices. The initial
complete V0 was explicitly requested; further changes should remain scoped to
the user's requested component rather than rewriting the project.

- Compare science and code: definitions, units, assumptions and implementation.
- Prefer simple, readable modules and transparent parameters over black boxes.
- Keep scientific processing separate from I/O, Qt state and display settings.
- Bacterial AUTO linking is distance-only. Do not add VAC or motion prediction.
- Only the user changes AUTO/MANUAL mode. Failure pauses but stays in AUTO.
- Manual selection bypasses AUTO size limits but stores a segmented centroid.
- Keep one point per frame, with manual/AUTO provenance and per-point settings.
- Do not fill gaps with predicted coordinates or draw them as measured continuity.
- Preserve reproducibility through parameters, video metadata, dependencies and logs.
- Do not claim biological accuracy from uninterrupted linking or visual smoothness.
- Review representative video frames and run meaningful tests for changed behavior.
