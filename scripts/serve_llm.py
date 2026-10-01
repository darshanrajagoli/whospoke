"""Start the local Stage-4 LLM server: AI4Bharat Airavata (7B), 4-bit, on http://127.0.0.1:8080/v1.

    python scripts/serve_llm.py                     # CPU; the first run builds the 4-bit model (see below)
    python scripts/serve_llm.py --gpu-layers 20     # part of the model on the GPU (all 32 layers + cache need ~6 GB)
    python scripts/serve_llm.py --prepare-only      # build llama-server and the 4-bit model file, then exit

The server is llama.cpp's ``llama-server`` (an OpenAI-compatible HTTP server). It is taken from ``--llama-bin``,
else from PATH (a llama.cpp release, ``brew install llama.cpp`` or ``winget install llama.cpp``), else built
once from the llama.cpp sources bundled in the ``llama-cpp-python`` package on PyPI (needs CMake and a C++
compiler).

Model file (``models/airavata-q4_k_m.gguf``, ~4 GB): ``--gguf`` uses an existing file. Otherwise the first run
downloads the 16-bit GGUF that ``ai4bharat/Airavata`` publishes (``Airavata.gguf``, 13.7 GB) and quantises it to
4 bits (Q4_K_M) with ``llama-quantize``, then deletes the 16-bit file. That needs ~18 GB of free disk once. The
repository is gated: accept its terms on huggingface.co, then set ``HF_TOKEN`` or run ``huggingface-cli login``.
(Its tokenizer is the same as ``tokenizer.model`` in the repository: same pieces, same scores.)

The chat format is Airavata's own (``<|system|>`` / ``<|user|>`` / ``<|assistant|>``, from its model card), passed
as a template file so it does not depend on what the GGUF metadata contains. Stage 4 sends JSON schemas, which
llama-server turns into grammars: the model can only produce replies of the expected shape.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
LLAMA_DIR = MODELS / "llama.cpp"                       # llama.cpp sources + build, when built here
TEMPLATE = PROJECT / "src" / "whospoke" / "resources" / "airavata_chat_template.jinja"
HF_REPO = "ai4bharat/Airavata"
HF_GGUF = "Airavata.gguf"                              # 16-bit GGUF published in the same repository
GGUF = MODELS / "airavata-q4_k_m.gguf"
EXE = ".exe" if os.name == "nt" else ""


def run(cmd: list, **kw) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_tool(name: str, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return (p / f"{name}{EXE}") if p.is_dir() else p
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    built = LLAMA_DIR / "build" / "bin" / f"{name}{EXE}"
    return built if built.exists() else None


def llama_sources() -> Path:
    """llama.cpp sources, unpacked from the llama-cpp-python source package on PyPI (it bundles them)."""
    if (LLAMA_DIR / "CMakeLists.txt").exists():
        return LLAMA_DIR
    MODELS.mkdir(parents=True, exist_ok=True)
    dl = MODELS / "_sdist"
    run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:", "llama-cpp-python==0.3.36",
         "-d", dl])
    sdist = next(dl.glob("llama_cpp_python-*.tar.gz"))
    with tarfile.open(sdist) as tar:
        root = tar.getnames()[0].split("/")[0]
        members = [m for m in tar.getmembers() if m.name.startswith(f"{root}/vendor/llama.cpp/")]
        tar.extractall(dl, members=members)
    shutil.move(str(dl / root / "vendor" / "llama.cpp"), LLAMA_DIR)
    shutil.rmtree(dl)
    return LLAMA_DIR


def build_tools(gpu: str | None) -> None:
    src = llama_sources()
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"]
    if gpu == "cuda":
        flags.append("-DGGML_CUDA=ON")
    run(["cmake", "-S", src, "-B", src / "build", *flags])
    run(["cmake", "--build", src / "build", "--config", "Release", "-j", str(os.cpu_count() or 4),
         "--target", "llama-server", "llama-quantize"])


def prepare_model(quantize_bin: Path, keep: bool) -> None:
    """ai4bharat/Airavata's own 16-bit GGUF (Hugging Face) → 4-bit Q4_K_M. No Python conversion step is needed."""
    from huggingface_hub import hf_hub_download

    MODELS.mkdir(parents=True, exist_ok=True)
    # the repository is gated: the token comes from HF_TOKEN, else from `huggingface-cli login`
    f16 = Path(hf_hub_download(HF_REPO, HF_GGUF, local_dir=MODELS / "Airavata-hf", token=os.getenv("HF_TOKEN") or None))
    run([quantize_bin, f16, GGUF, "Q4_K_M"])
    if not keep:
        shutil.rmtree(MODELS / "Airavata-hf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gguf", default=None, help=f"model file to serve (default: {GGUF.relative_to(PROJECT)})")
    ap.add_argument("--llama-bin", default=None, help="llama-server executable, or the folder that holds it")
    ap.add_argument("--build", choices=["cpu", "cuda"], default=None,
                    help="(re)build llama-server from source, for CPU or with CUDA")
    ap.add_argument("--gpu-layers", type=int, default=0, help="model layers on the GPU (0 = CPU only, 99 = all)")
    ap.add_argument("--ctx", type=int, default=4096, help="context window in tokens (Airavata: 4096)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: llama.cpp's choice)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--prepare-only", action="store_true", help="build the model file and exit")
    ap.add_argument("--keep-intermediate", action="store_true", help="keep the downloaded 16-bit GGUF")
    args = ap.parse_args()

    if args.build:
        build_tools(args.build)
    server = find_tool("llama-server", args.llama_bin)
    if server is None:
        build_tools(None)
        server = find_tool("llama-server", None)
    gguf = Path(args.gguf) if args.gguf else GGUF
    if not gguf.exists():
        if args.gguf:
            raise SystemExit(f"model file not found: {gguf}")
        quantize = find_tool("llama-quantize", str(server.parent))
        if not quantize.exists():
            build_tools(None)
            quantize = find_tool("llama-quantize", None)
        prepare_model(quantize, args.keep_intermediate)
    if args.prepare_only:
        print(f"ready: {gguf}")
        return
    cmd = [server, "-m", gguf, "--host", args.host, "--port", str(args.port), "-c", str(args.ctx),
           "-ngl", str(args.gpu_layers), "--jinja", "--chat-template-file", TEMPLATE, "--alias", "ai4bharat/Airavata"]
    if args.threads:
        cmd += ["-t", str(args.threads)]
    print(f"Stage-4 LLM server: http://{args.host}:{args.port}/v1  (Ctrl+C to stop)", flush=True)
    run(cmd)


if __name__ == "__main__":
    main()
