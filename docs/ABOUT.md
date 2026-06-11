# About this project

This app was developed off excellent work by Chen-zexi on
[vllm-cli](https://github.com/Chen-zexi/vllm-cli): many thanks for the CLI
version, which the author recommends for use when VRAM (your GPU's memory) is
tight — booting into a TTY without running a display server can free up over
1 GB of VRAM.

The idea behind this project was the frustration that comes with attempting
to squeeze LLMs of various types onto individual hardware. The flags are
cryptic, loading fails frequently, and documentation is not always helpful.
This is an effort to help the local LLM community more easily run models. The
app carefully measures available VRAM, makes suggestions, and can adjust
settings to allow a model to be run without hours of trial and error. Or at
least, fewer hours of trial and error.

## Why a GUI, and why both vLLM *and* llama.cpp?

vLLM and llama.cpp solve overlapping but different problems:

- **vLLM** is built for throughput — serving many requests at once, with
  HuggingFace-format (`.safetensors`) models. It's the engine of choice when
  you have one or more dedicated NVIDIA GPUs and want an OpenAI-compatible
  server that can handle real load.
- **llama.cpp** is built for *reach* — it runs almost everywhere (CPU, NVIDIA,
  AMD, Apple Silicon via Metal), loads `.gguf` files (a single-file,
  pre-quantized format), and is usually the easiest on-ramp for someone with a
  laptop or a single consumer GPU.

A newcomer doesn't know which of these fits their situation, and the flags for
*either* engine assume you already understand GPU memory budgeting, KV-cache
math, tensor parallelism, and quantization formats. This project's job is to
stand between you and that complexity: detect your hardware, recommend an
engine, and translate every flag into "what does this do, and is it safe on
*your* machine, for *this* model, right now?"

## A note on "fewer hours," not "zero hours"

Local LLM serving is still a fast-moving target — engines change their flags
between releases (we hit this ourselves: a recent vLLM release removed
`--swap-space`, which broke a hardcoded command), new model architectures
appear faster than engines can support them, and consumer GPU memory is
genuinely tight. This app's traffic-light system and memory gauges are
*estimates and guardrails*, not guarantees. When something still goes wrong,
the goal is that the error message in this app is in plain English instead of
a 70-line Python traceback — and that fixing it takes one slider adjustment
instead of an hour of searching GitHub issues.

## Built with Claude Fable 5

This entire application — backend, frontend, advisor logic, tests, and this
documentation — was built in collaboration with **Claude Fable 5**
(Anthropic's Claude Code). From cloning and reviewing vllm-cli, through
designing the hardware-aware advisor engine, building the React frontend, to
diagnosing real failed launches on the developer's actual hardware (two RTX
5060 Ti GPUs) and shipping fixes for them — the whole loop of *spec → code →
test → run it for real → read the failure → fix it* happened in conversation
with Claude.

A genuine shoutout to **Anthropic** for Claude Code and the underlying models.
The ability to go from "clone this repo and review it" to "here's a working,
tested, end-to-end GUI that launched a real model on real GPUs" in a single
session — including catching and fixing its own bugs against real hardware
failures — is something else. Thank you.

## License and credit

This project is MIT-licensed. Portions of the configuration model, flag
catalog concept, server lifecycle handling, and model discovery approach are
adapted from [vllm-cli](https://github.com/Chen-zexi/vllm-cli) by **Chen-zexi**
(also MIT). See [VLLM_CLI_BACKGROUND.md](VLLM_CLI_BACKGROUND.md) for a
breakdown of what was kept, what was changed, and what's new.
