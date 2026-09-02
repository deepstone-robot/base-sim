#!/usr/bin/env bash
# Run mjlab training on CPU for pipeline smoke-testing.
#
# Two macOS-specific knobs are set BEFORE any python import so that mjlab's
# own training entry point (mjlab/scripts/train.py) is happy:
#
#   1. CUDA_VISIBLE_DEVICES=""  →  mjlab.utils.gpu.select_gpus() returns
#      (None, 0) ("CPU mode" per the documented behavior in gpu.py:41-42)
#      instead of indexing into an empty torch.cuda.device_count() list
#      (gpu.py:63-70), which is the IndexError microduck_rl's pyproject
#      comments warn about on aarch64.
#
#   2. MUJOCO_GL=<platform>  →  mjlab/scripts/train.py:199 unconditionally
#      sets MUJOCO_GL=egl at runtime, but mujoco's import-time gl_context
#      init (mujoco/rendering/classic/gl_context.py) rejects "egl" on
#      Darwin. Use cgl on macOS, egl on Linux (the GPU production path),
#      and leave it default elsewhere.
#
# GPU validation on a separate Linux/CUDA box is the default:
#      uv run train <TASK> [args...]
# This wrapper is ONLY for CPU smoke-tests; the GPU code path in pyproject
# (aarch64-linux torch source, mjlab/warp CUDA) is untouched.
#
# Usage:
#   scripts/run_cpu.sh <TASK> [extra args passed to mjlab train...]
#
#   scripts/run_cpu.sh Mjlab-Cartpole-Balance \
#     --env.scene.num-envs 4 --agent.max-iterations 1

set -euo pipefail

case "$(uname -s)" in
  Linux)   gl="egl" ;;
  Darwin)  gl="cgl" ;;
  *)       gl="" ;;  # leave default; covers Windows, etc.
esac

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL="$gl"
# rsl_rl's WandbSummaryWriter calls wandb.init() unconditionally and prompts
# for an API key on first use. Disable wandb entirely for smoke tests; the
# GPU path uses `uv run train …` directly with wandb enabled.
export WANDB_MODE=disabled

exec uv run train "$@"