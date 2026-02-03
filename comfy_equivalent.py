import io
import logging
import os
from collections.abc import Iterator

import numpy as np
import torch
from PIL import Image

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    decode_video as vae_decode_video,
    get_video_chunks_number,
)
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    get_device,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

LOGGER = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_clamped(name: str, default: int, *, min_value: int, max_value: int) -> int:
    value = os.getenv(name)
    if value is None:
        return max(min_value, min(default, max_value))
    try:
        parsed = int(float(value))
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s; using %s", name, value, default)
        parsed = default
    return max(min_value, min(parsed, max_value))


def log_cuda_mem(tag: str) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    max_allocated = int(torch.cuda.max_memory_allocated())
    max_reserved = int(torch.cuda.max_memory_reserved())
    LOGGER.info(
        "CUDA_MEM %s allocated=%s reserved=%s max_allocated=%s max_reserved=%s",
        tag,
        allocated,
        reserved,
        max_allocated,
        max_reserved,
    )
    if os.getenv("MEM_SUMMARY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        LOGGER.info("CUDA_MEM_SUMMARY %s\n%s", tag, torch.cuda.memory_summary())
    return allocated, reserved


def _wrap_denoise_with_mem(
    denoise_fn,
    *,
    log_every: int | None,
    log_prefix: str | None,
):
    if not log_every:
        return denoise_fn

    def wrapped(video_state, audio_state, sigmas, step_index):
        if step_index % log_every == 0:
            tag = f"chunk={log_prefix} step={step_index}" if log_prefix is not None else f"step={step_index}"
            mem = log_cuda_mem(tag)
            if mem is not None and log_prefix is not None:
                LOGGER.info(
                    "chunk=%s step=%s allocated=%s reserved=%s",
                    log_prefix,
                    step_index,
                    mem[0],
                    mem[1],
                )
        return denoise_fn(video_state, audio_state, sigmas, step_index)

    return wrapped


def _tensor_frames_to_pil(video_iter: torch.Tensor | Iterator[torch.Tensor]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    if isinstance(video_iter, torch.Tensor):
        video_iter = iter([video_iter])
    for chunk in video_iter:
        chunk_cpu = chunk.detach().to("cpu")
        if chunk_cpu.dtype != torch.uint8:
            chunk_cpu = torch.clamp(chunk_cpu, 0, 255).to(torch.uint8)
        for frame in chunk_cpu:
            frames.append(Image.fromarray(frame.numpy(), mode="RGB"))
    return frames


def _audio_tensor_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio_cpu = audio.detach().to("cpu")
    if audio_cpu.ndim == 1:
        audio_cpu = audio_cpu[:, None]
    if audio_cpu.shape[1] != 2 and audio_cpu.shape[0] == 2:
        audio_cpu = audio_cpu.T
    if audio_cpu.shape[1] != 2:
        raise ValueError(f"Expected stereo audio with 2 channels, got shape {audio_cpu.shape}.")
    audio_cpu = torch.clamp(audio_cpu, -1.0, 1.0)
    audio_i16 = (audio_cpu * 32767.0).to(torch.int16).numpy()

    import wave  # local import to keep module light

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_i16.tobytes())
    return buffer.getvalue()


def render_comfy_equivalent(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    ckpt_path: str,
    gemma_root: str,
    distilled_lora_path: str,
    distilled_lora_strength: float,
    spatial_upscaler_path: str,
    seed_stage1: int = 10,
    seed_stage2: int = 0,
    chunk_index: int | None = None,
) -> tuple[list[Image.Image], bytes | None, torch.Tensor | None]:
    if (num_frames - 1) % 8 != 0:
        raise ValueError(f"num_frames must be 1 + 8*k (got {num_frames})")

    assert_resolution(height=height, width=width, is_two_stage=True)

    LOGGER.info(
        "Comfy-equivalent params: width=%s height=%s fps=%s num_frames=%s seeds=%s/%s sigmas=%s",
        width,
        height,
        fps,
        num_frames,
        seed_stage1,
        seed_stage2,
        "0.909375,0.725,0.421875,0.0",
    )

    device = get_device()
    dtype = torch.bfloat16
    fp8 = _env_bool("LTX2_ENABLE_FP8", False)
    log_every = int(os.getenv("LTX2_MEM_LOG_EVERY", "5"))
    if log_every <= 0:
        log_every = None

    log_cuda_mem(f"chunk={chunk_index} start")

    base_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt_path,
        spatial_upsampler_path=spatial_upscaler_path,
        gemma_root_path=gemma_root,
        loras=(),
        fp8transformer=fp8,
    )

    stage2_loras: tuple[LoraPathStrengthAndSDOps, ...] = ()
    if distilled_lora_path and _env_bool("LTX2_USE_DISTILLED", False):
        stage2_loras = (
            LoraPathStrengthAndSDOps(distilled_lora_path, distilled_lora_strength, LTXV_LORA_COMFY_RENAMING_MAP),
        )
    elif distilled_lora_path:
        LOGGER.info("Comfy-equivalent: ignoring distilled LoRA (LTX2_USE_DISTILLED=0).")
    stage2_ledger = base_ledger.with_loras(loras=stage2_loras) if stage2_loras else base_ledger

    pipeline_components = PipelineComponents(dtype=dtype, device=device)

    text_encoder = base_ledger.text_encoder()
    context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
    v_context_p, a_context_p = context_p
    v_context_n, a_context_n = context_n

    torch.cuda.synchronize()
    del text_encoder
    cleanup_memory()

    stage_w = width // 2
    stage_h = height // 2
    LOGGER.info("Comfy-equivalent stage size: stage_w=%s stage_h=%s", stage_w, stage_h)

    generator_stage1 = torch.Generator(device=device).manual_seed(int(seed_stage1))
    noiser_stage1 = GaussianNoiser(generator=generator_stage1)
    stepper = EulerDiffusionStep()

    video_encoder = base_ledger.video_encoder()
    transformer_stage1 = base_ledger.transformer()
    scheduler = LTX2Scheduler()
    stage1_steps = _env_int_clamped("LTX2_COMMERCIAL_STEPS", _env_int_clamped("LTX2_STAGE1_STEPS", 20, min_value=1, max_value=200), min_value=1, max_value=200)
    LOGGER.info("Comfy-equivalent stage1 steps=%s", stage1_steps)
    sigmas = scheduler.execute(
        steps=stage1_steps,
        max_shift=2.05,
        base_shift=0.95,
        stretch=True,
        terminal=0.1,
    ).to(dtype=torch.float32, device=device)

    def stage1_denoising_loop(
        sigmas_tensor: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        stepper_impl: EulerDiffusionStep,
    ) -> tuple[LatentState, LatentState]:
        return euler_denoising_loop(
            sigmas=sigmas_tensor,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper_impl,
            denoise_fn=_wrap_denoise_with_mem(
                multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=MultiModalGuiderParams(cfg_scale=4.0),
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=MultiModalGuiderParams(cfg_scale=4.0),
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer_stage1,
                ),
                log_every=log_every,
                log_prefix=str(chunk_index) if chunk_index is not None else None,
            ),
        )

    stage1_output_shape = VideoPixelShape(
        batch=1,
        frames=num_frames,
        width=stage_w,
        height=stage_h,
        fps=float(fps),
    )
    stage1_conditionings = image_conditionings_by_replacing_latent(
        images=[],
        height=stage1_output_shape.height,
        width=stage1_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )

    log_cuda_mem(f"chunk={chunk_index} before_stage1")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        video_state, audio_state = denoise_audio_video(
            output_shape=stage1_output_shape,
            conditionings=stage1_conditionings,
            noiser=noiser_stage1,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=stage1_denoising_loop,
            components=pipeline_components,
            dtype=dtype,
            device=device,
        )
    log_cuda_mem(f"chunk={chunk_index} after_stage1")

    torch.cuda.synchronize()
    del transformer_stage1
    cleanup_memory()

    upscaled_video_latent = upsample_video(
        latent=video_state.latent[:1],
        video_encoder=video_encoder,
        upsampler=stage2_ledger.spatial_upsampler(),
    )

    torch.cuda.synchronize()
    cleanup_memory()

    generator_stage2 = torch.Generator(device=device).manual_seed(int(seed_stage2))
    noiser_stage2 = GaussianNoiser(generator=generator_stage2)

    transformer_stage2 = stage2_ledger.transformer()
    stage2_sigmas = torch.tensor([0.909375, 0.725, 0.421875, 0.0], device=device, dtype=torch.float32)

    def stage2_denoising_loop(
        sigmas_tensor: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        stepper_impl: EulerDiffusionStep,
    ) -> tuple[LatentState, LatentState]:
        return euler_denoising_loop(
            sigmas=sigmas_tensor,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper_impl,
            denoise_fn=_wrap_denoise_with_mem(
                simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer_stage2,
                ),
                log_every=log_every,
                log_prefix=str(chunk_index) if chunk_index is not None else None,
            ),
        )

    stage2_output_shape = VideoPixelShape(
        batch=1,
        frames=num_frames,
        width=width,
        height=height,
        fps=float(fps),
    )
    stage2_conditionings = image_conditionings_by_replacing_latent(
        images=[],
        height=stage2_output_shape.height,
        width=stage2_output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=device,
    )

    log_cuda_mem(f"chunk={chunk_index} before_stage2")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        video_state, audio_state = denoise_audio_video(
            output_shape=stage2_output_shape,
            conditionings=stage2_conditionings,
            noiser=noiser_stage2,
            sigmas=stage2_sigmas,
            stepper=stepper,
            denoising_loop_fn=stage2_denoising_loop,
            components=pipeline_components,
            dtype=dtype,
            device=device,
            noise_scale=stage2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )
    log_cuda_mem(f"chunk={chunk_index} after_stage2")

    torch.cuda.synchronize()
    del transformer_stage2
    del video_encoder
    cleanup_memory()

    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=4096, tile_overlap_in_frames=8),
    )

    decoded_video = vae_decode_video(video_state.latent, base_ledger.video_decoder(), tiling_config, generator_stage2)
    decoded_audio = vae_decode_audio(audio_state.latent, base_ledger.audio_decoder(), base_ledger.vocoder())

    frames = _tensor_frames_to_pil(decoded_video)
    decoded_audio_cpu = decoded_audio.detach().to("cpu") if decoded_audio is not None else None
    wav_bytes = _audio_tensor_to_wav_bytes(decoded_audio_cpu, AUDIO_SAMPLE_RATE) if decoded_audio_cpu is not None else None

    log_cuda_mem(f"chunk={chunk_index} end")

    del video_state, audio_state, sigmas, stage2_sigmas, upscaled_video_latent, decoded_video, decoded_audio
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log_cuda_mem(f"chunk={chunk_index} after_cleanup")

    return frames, wav_bytes, decoded_audio_cpu


def render_comfy_equivalent_mp4(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    ckpt_path: str,
    gemma_root: str,
    distilled_lora_path: str,
    distilled_lora_strength: float,
    spatial_upscaler_path: str,
    seed_stage1: int = 10,
    seed_stage2: int = 0,
    out_mp4_path: str = "/tmp/ltx_comfy_equiv.mp4",
    chunk_index: int | None = None,
) -> str:
    frames, _wav_bytes, audio_tensor = render_comfy_equivalent(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        ckpt_path=ckpt_path,
        gemma_root=gemma_root,
        distilled_lora_path=distilled_lora_path,
        distilled_lora_strength=distilled_lora_strength,
        spatial_upscaler_path=spatial_upscaler_path,
        seed_stage1=seed_stage1,
        seed_stage2=seed_stage2,
        chunk_index=chunk_index,
    )

    video_tensor = torch.stack([torch.from_numpy(np.array(frame)) for frame in frames])
    chunks_number = get_video_chunks_number(num_frames, TilingConfig.default())
    encode_video(
        video=video_tensor,
        fps=fps,
        audio=audio_tensor,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=out_mp4_path,
        video_chunks_number=chunks_number,
    )
    return out_mp4_path
