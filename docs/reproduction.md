# Reproduction guide

The scripts were exported from the completed study. Packaging checks covered syntax, stored-result arithmetic, source hashes, and command rendering. The portable export has not been run as a fresh end-to-end GPU experiment. Python 3 standard-library modules are enough to audit the stored data.

## Build the runtime

Use an ARM64 builder and the pinned Mjolnir tree as the build context. The included Dockerfile is the upstream file with one local change: `NVIDIA_VISIBLE_DEVICES=void` after `FROM`, so build probes do not use a GPU. It includes the upstream patch application and CPU build checks.

```sh
git clone https://github.com/thor-arena/mjolnir.git
git -C mjolnir checkout 171d9ce692324cb740342e39d53dd95bdaf0c441
docker build --platform linux/arm64 \
  -f "$PWD/build/Dockerfile" \
  --build-arg MJOLNIR_REVISION=171d9ce692324cb740342e39d53dd95bdaf0c441 \
  -t thor-dflash-study:local ./mjolnir
```

The base image digest is pinned in the Dockerfile and `versions.json`. The recorded local image ID is provenance only; it is not a public image to pull. Upstream build steps can resolve external packages. A new build is not guaranteed to have the same image bytes. Keep upstream patch verification enabled and inspect any version drift.

## Obtain model snapshots

Download the full target snapshot and one full drafter snapshot from the repositories and exact revisions in `versions.json`. For example, with `huggingface_hub` installed:

```python
import json
from huggingface_hub import snapshot_download
versions = json.load(open('versions.json'))
for key, directory in [('target', 'models/target'), ('nvfp4_drafter', 'models/drafter')]:
    pin = versions[key]
    snapshot_download(repo_id=pin['repo'], revision=pin['revision'], local_dir=directory)
```

For the BF16 trial, use `bf16_drafter`. Downloading weights and running the server require enough disk and unified memory. The study used Thor SM110, CUDA 13.0, and an R580 host driver. It did not test Spark SM121 or RTX SM120 as substitutes.

## Render a serving command

`configs/*.json` retain the measured Docker arguments, with private paths, GPU UUID, and image ID replaced by named placeholders. Set these values for your system. `render_command.py` prints a shell-quoted command; it does not run it.

```sh
export REPO_DIR="$PWD"
export MJOLNIR_DIR="$PWD/mjolnir"
export TARGET_DIR="$PWD/models/target"
export DRAFTER_DIR="$PWD/models/drafter"
export CACHE_DIR="$PWD/.cache"
export IMAGE=thor-dflash-study:local
export GPU_UUID='GPU-REPLACE-WITH-YOUR-THOR-UUID'
mkdir -p "$CACHE_DIR"
python3 scripts/render_command.py configs/mjolnir-dflash2-nvfp4-7-hnd-kv3.json
```

Review the printed command before executing it. The runtime must support the NVIDIA CDI device selector in the template. `launch.py` requires exactly one visible GPU and capability 11.0. The explicit GPU environment values replace the build-time `void` setting. The templates bind the API to localhost port 18081 and use a 48 GiB memory limit. They retain fixed container names; stop and remove a completed test container before using its name again.

The six DFlash2 templates use the included adapter or unmodified draft head as appropriate. Chisel variants mount `chisel/` at `/opt/chisel`; its `dist-info` entry point registers the plugin through `PYTHONPATH`. The full-vocabulary FP8 control uses the identity token list. The reduced cases use the pinned 100,032-token list.

The MTP template is an archival command record. It also needs `ARCHIVED_WEIGHT_UTILS` and `ARCHIVED_MTP_VOCAB`, which are not included in this DFlash2 export. The MTP row can be audited from stored data but is not offered as a complete runnable reproduction. Its 64K list and loading customization must not be guessed from the DFlash2 setup.

## Run the client scripts for a future experiment

After starting a server and waiting for readiness, choose a new output directory outside `results/`:

```sh
python3 scripts/client_speed_repeated.py runs/example
python3 scripts/client_quality.py runs/example
```

`BENCHMARK_URL` can change the default `http://127.0.0.1:18081`. The served model name remains `qwen27-test`. Client changes for this export only add output-directory creation, an optional URL override, and support for absent host-guard event files. Prompts, timing logic, and quality checks are retained. The speed script writes metrics after each request. Review new raw metrics before publishing them; they can contain deployment labels.

`verify_code.py` executes generated Python after limited AST checks. Those checks are not a security sandbox. The original code validation ran in a separate CPU container. For future use, run it in an isolated container with no network, read-only inputs, memory/CPU limits, and a timeout. Stored `rawcode-verification.json` files contain the original outcomes. The offline audit reads these outcomes without executing the generated code.

The private service guard, mining controls, SSH scripts, and host monitoring are deliberately absent. The export does not reproduce their lifecycle behavior. See the OOM and coexistence limits in [method.md](method.md) before planning any shared-host experiment.
