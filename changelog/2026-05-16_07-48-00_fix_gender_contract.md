# API Contract Alignment: Gender Predictions
Date: 2026-05-16

## What was done
- Modified `app/inference/pipeline.py` to map `child` gender predictions to `unknown`.
- Updated `_parse_gender` logic to strictly follow the "Expected API Contract" defined in the assignment (`male | female | unknown`).

## Why it was done
- The assignment's specified contract does not include a `child` category. Mapping internal `child` classifications (which can occur with the wav2vec2 model) to `unknown` ensures the service is compatible with the interviewer's expected schema and automated evaluation tools.
