# LTX-2 vendored primitives map

This document points to vendored LTX-2 primitives to reference (do not rewrite).

## Text encoding / prompt embeddings
- `vendor/ltx2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py`
  - `encode_text(...)`
  - `GemmaTextEncoderModelBase` (prompt preprocessing + embeddings)
- `vendor/ltx2/packages/ltx-core/src/ltx_core/text_encoders/gemma/__init__.py`
  - re-exports `encode_text`, `AVGemmaTextEncoderModel`, `VideoGemmaTextEncoderModel`
- Usage in pipelines:
  - `vendor/ltx2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py`
  - `vendor/ltx2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py`

## Scheduler / sigmas / shift logic
- `vendor/ltx2/packages/ltx-core/src/ltx_core/components/schedulers.py`
  - `LTX2Scheduler.execute(...)` (sigma schedule + shift/stretches)
  - `flux_time_shift(...)`

## Euler ancestral sampler stepper
- `vendor/ltx2/packages/ltx-core/src/ltx_core/components/diffusion_steps.py`
  - `EulerDiffusionStep.step(...)`

## Latent spatial upsample x2
- `vendor/ltx2/packages/ltx-core/src/ltx_core/model/upsampler/model.py`
  - `LatentUpsampler`
  - `upsample_video(...)`

## Tiled VAE decode
- `vendor/ltx2/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py`
  - `VideoDecoder.tiled_decode(...)`
- Example usage in validation:
  - `vendor/ltx2/packages/ltx-trainer/src/ltx_trainer/validation_sampler.py`

## Audio decode
- `vendor/ltx2/packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py`
  - `AudioDecoder`
  - `decode_audio(...)`
- Vocoder implementation:
  - `vendor/ltx2/packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py`
