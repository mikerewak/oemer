import os
import pickle
import platform
import subprocess
import threading
from functools import lru_cache
from PIL import Image

import cv2
import numpy as np

from oemer import MODULE_PATH

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

_MACHINE = platform.machine().lower()
_ARCH = (
    "arm64" if _MACHINE in {"arm64", "aarch64"}
    else "x86_64" if _MACHINE in {"amd64", "x86_64"}
    else _MACHINE
)
_IS_DARWIN = platform.system() == "Darwin"
_IS_LINUX = platform.system() == "Linux"


def _machine_arch():
    return _ARCH


def _sysctl_int(key):
    """macOS sysctl integer query."""
    if not _IS_DARWIN:
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2
        )
        return int(result.stdout.strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _perf_cores():
    if _IS_DARWIN and _ARCH == "arm64":
        return _sysctl_int("hw.perflevel0.physicalcpu") or 0
    return 0


@lru_cache(maxsize=1)
def _eff_cores():
    if _IS_DARWIN and _ARCH == "arm64":
        return _sysctl_int("hw.perflevel1.physicalcpu") or 0
    return 0


@lru_cache(maxsize=1)
def _linux_physical_cores():
    """Physical cores on Linux (socket * cores), ignoring HT siblings."""
    if not _IS_LINUX:
        return 0
    try:
        out = subprocess.run(
            ["lscpu", "--parse=core,socket"],
            capture_output=True, text=True, timeout=3,
        )
        cores = {
            tuple(int(x) for x in line.split(",")[:2])
            for line in out.stdout.strip().splitlines()
            if line and not line.startswith("#")
        }
        if cores:
            return len(cores)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            siblings = 0
            for line in fh:
                if line.startswith("siblings"):
                    try:
                        siblings = max(siblings, int(line.split(":")[1].strip()))
                    except ValueError:
                        pass
            if siblings:
                return max(1, (os.cpu_count() or 1) // max(1, siblings))
    except Exception:
        pass
    return 0


@lru_cache(maxsize=1)
def _x86_isa_flags():
    """Detect AVX2/AVX-512 support."""
    if _ARCH != "x86_64":
        return frozenset()
    flags = set()
    if _IS_LINUX:
        try:
            out = subprocess.run(
                ["grep", "-m1", "flags", "/proc/cpuinfo"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                parts = out.stdout.strip().split(":")
                if len(parts) >= 2:
                    flags = set(parts[-1].strip().split())
        except Exception:
            pass
    if _IS_DARWIN:
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output=True, text=True, timeout=2,
            )
            for line in out.stdout.strip().splitlines():
                flags.update(line.lower().split())
        except Exception:
            pass
    return frozenset(flags)


def _optimal_intra_threads():
    """Best intra-op thread count for current CPU topology."""
    if _IS_DARWIN and _ARCH == "arm64":
        return max(1, _perf_cores() or 4)
    if _ARCH == "x86_64":
        phys = _linux_physical_cores()
        if phys:
            return phys
    return max(2, (os.cpu_count() or 4) // 2)


def _optimal_inter_threads():
    """Best inter-op thread count (keep low to avoid contention)."""
    if _IS_DARWIN and _ARCH == "arm64":
        return min(_eff_cores() or 2, 2)
    return 1


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Thread configuration (OpenCV + BLAS)
# ---------------------------------------------------------------------------

_CV2_THREADS_CONFIGURED = False
_CV2_THREADS_LOCK = threading.Lock()


def _configure_cv2_threads():
    global _CV2_THREADS_CONFIGURED
    if _CV2_THREADS_CONFIGURED:
        return
    with _CV2_THREADS_LOCK:
        if _CV2_THREADS_CONFIGURED:
            return
        default_threads = 1 if _IS_DARWIN and _ARCH == "arm64" else 0
        threads = _env_int("OEMER_OPENCV_THREADS", default_threads)
        if threads > 0:
            cv2.setNumThreads(threads)
        _CV2_THREADS_CONFIGURED = True


_BLAS_THREADS_CONFIGURED = False
_BLAS_THREADS_LOCK = threading.Lock()


def _configure_blas_threads():
    """Pin BLAS/MKL/OpenMP thread counts to avoid over-subscription.

    Ensures numpy, scipy, and any system BLAS library use the same
    intra-op thread count derived from CPU topology.  Must be called
    before any heavy numpy / model inference work.
    """
    global _BLAS_THREADS_CONFIGURED
    if _BLAS_THREADS_CONFIGURED:
        return
    with _BLAS_THREADS_LOCK:
        if _BLAS_THREADS_CONFIGURED:
            return
        nt = str(_optimal_intra_threads())
        os.environ.setdefault("OMP_NUM_THREADS", nt)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", nt)
        os.environ.setdefault("MKL_NUM_THREADS", nt)
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", nt)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", nt)
        os.environ.setdefault("BLIS_NUM_THREADS", nt)

        # Darwin: passive spin-wait so helper threads don't burn CPU.
        if _IS_DARWIN:
            os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

        # x86_64 Linux: MKL-specific affinity for dense matmul throughput.
        if _IS_LINUX and _ARCH == "x86_64":
            os.environ.setdefault("KMP_BLOCKTIME", "0")
            os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

        _BLAS_THREADS_CONFIGURED = True


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_image(image: Image):
    # Estimate target size with number of pixels.
    # Best number would be 3M~4.35M pixels.
    w, h = image.size
    pis = w * h
    if 3000000 <= pis <= 4350000:
        return image
    lb = 3000000 / pis
    ub = 4350000 / pis
    ratio = pow((lb + ub) / 2, 0.5)
    tar_w = round(ratio * w)
    tar_h = round(ratio * h)
    print(tar_w, tar_h)
    return image.resize((tar_w, tar_h))


def _target_size(w, h):
    pixels = w * h
    if 3000000 <= pixels <= 4350000:
        return w, h
    lb = 3000000 / pixels
    ub = 4350000 / pixels
    ratio = pow((lb + ub) / 2, 0.5)
    return round(ratio * w), round(ratio * h)


def _load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")
    # BGR->RGB via slice-reversal + copy; faster than cv2.cvtColor on many
    # OpenCV builds because it avoids the colorspace LUT lookup.
    image = image[..., ::-1].copy()
    target_w, target_h = _target_size(image.shape[1], image.shape[0])
    if target_w != image.shape[1] or target_h != image.shape[0]:
        interpolation = cv2.INTER_AREA if target_w * target_h < image.shape[0] * image.shape[1] else cv2.INTER_CUBIC
        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        print(target_w, target_h)
    # Avoid the copy when the array is already C-contiguous.
    if not image.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(image)
    return image


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------

def _onnx_provider_names():
    requested = (
        os.environ.get("OEMER_ONNXRUNTIME_PROVIDER")
        or os.environ.get("OMR_OEMER_ONNXRUNTIME_PROVIDER")
        or ""
    ).strip().lower()
    provider_aliases = {
        "cpu": "CPUExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "gpu": "CUDAExecutionProvider",
    }
    if requested:
        if requested not in provider_aliases:
            raise RuntimeError(f"Unsupported OEMER_ONNXRUNTIME_PROVIDER={requested!r}")
        return (provider_aliases[requested], "CPUExecutionProvider")

    arch = _machine_arch()
    if _IS_DARWIN and arch == "arm64":
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    # Default to CPU everywhere else — the Sibelius adapter handles CUDA
    # selection separately via OMR_OEMER_ONNXRUNTIME_PROVIDER.
    return ("CPUExecutionProvider",)


def _onnx_session_options(opt_path=None):
    import onnxruntime as rt

    opts = rt.SessionOptions()
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    arch = _machine_arch()
    is_apple_silicon = _IS_DARWIN and arch == "arm64"

    intra_threads = _env_int("OEMER_ONNX_INTRA_OP_THREADS", 0)
    inter_threads = _env_int("OEMER_ONNX_INTER_OP_THREADS", 0)

    if is_apple_silicon:
        # CoreML EP manages its own parallelism (ANE/GPU); keep ORT
        # sequential to avoid contention.
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        if intra_threads == 0:
            intra_threads = 1
    else:
        # x86_64 / generic: run ops in parallel across physical cores.
        opts.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        if intra_threads == 0:
            intra_threads = _optimal_intra_threads()
        if inter_threads == 0:
            inter_threads = _optimal_inter_threads()

    if intra_threads > 0:
        opts.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        opts.inter_op_num_threads = inter_threads

    opts.enable_mem_pattern = os.environ.get("OEMER_ONNX_DISABLE_MEM_PATTERN") != "1"
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_reuse = True

    # Persist the ORT-optimized graph so subsequent cold starts skip
    # graph-optimization overhead.
    if opt_path:
        opts.optimized_model_filepath = opt_path

    return opts


def _provider_tag(provider):
    """Short lowercase label used in cache-file names (avoids
    cross-provider cache pollution)."""
    name = provider if isinstance(provider, str) else provider[0]
    return name.replace("ExecutionProvider", "").lower()


@lru_cache(maxsize=4)
def _load_onnx_session(model_path):
    import onnxruntime as rt

    onnx_path = os.path.join(model_path, "model.onnx")
    metadata = pickle.load(open(os.path.join(model_path, "metadata.pkl"), "rb"))
    requested_providers = _onnx_provider_names()
    available = set(rt.get_available_providers())
    explicit = bool(
        os.environ.get("OEMER_ONNXRUNTIME_PROVIDER")
        or os.environ.get("OMR_OEMER_ONNXRUNTIME_PROVIDER")
    )
    missing = [p for p in requested_providers if p not in available]
    if explicit and requested_providers[0] in missing:
        raise RuntimeError(
            f"Requested ONNXRuntime provider {requested_providers[0]} is unavailable; "
            f"available providers: {sorted(available)}"
        )

    # Build provider list; attach CoreML options to request ANE + GPU dispatch.
    providers = []
    for p in requested_providers:
        if p not in available:
            continue
        if p == "CoreMLExecutionProvider":
            providers.append((p, {"MLComputeUnits": "ALL"}))
        else:
            providers.append(p)
    if not providers:
        providers = ["CPUExecutionProvider"]

    # Per-provider optimized graph cache: avoids re-running graph
    # optimization on each cold start.
    tag = _provider_tag(providers[0])
    opt_path = os.path.join(model_path, f"model_opt_{tag}.onnx")
    if os.path.exists(opt_path):
        load_path, sess_opts = opt_path, _onnx_session_options()
    else:
        load_path, sess_opts = onnx_path, _onnx_session_options(opt_path)

    sess = rt.InferenceSession(load_path, sess_options=sess_opts, providers=providers)
    return sess, metadata


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _tile_positions(length, win_size, step_size):
    if length <= win_size:
        return [0]
    positions = list(range(0, length - win_size + 1, step_size))
    final = length - win_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def _tile_coordinates(height, width, win_size, step_size):
    y_positions = _tile_positions(height, win_size, step_size)
    x_positions = _tile_positions(width, win_size, step_size)
    return [(y, x) for y in y_positions for x in x_positions]


def inference(model_path, img_path, step_size=128, batch_size=16, manual_th=None, use_tf=False):
    _configure_cv2_threads()
    _configure_blas_threads()

    if use_tf:
        import tensorflow as tf

        arch_path = os.path.join(model_path, "arch.json")
        w_path = os.path.join(model_path, "weights.h5")
        model = tf.keras.models.model_from_json(open(arch_path, "r").read())
        model.load_weights(w_path)
        input_shape = model.input_shape
        output_shape = model.output_shape
    else:
        sess, metadata = _load_onnx_session(os.path.abspath(model_path))
        output_names = metadata['output_names']
        input_shape = metadata['input_shape']
        output_shape = metadata['output_shape']

    image = _load_image(img_path)
    win_size = input_shape[1]
    n_channels = image.shape[-1]
    batch_size = _env_int("OEMER_ONNX_BATCH_SIZE", batch_size)
    batch_size = max(1, batch_size)
    coords = _tile_coordinates(image.shape[0], image.shape[1], win_size, step_size)

    output_shape = image.shape[:2] + (output_shape[-1],)
    out = np.zeros(output_shape, dtype=np.float32)
    mask = np.zeros(output_shape, dtype=np.float32)

    # Pre-allocate batch buffer once and reuse across iterations.
    batch = np.empty((batch_size, win_size, win_size, n_channels), dtype=image.dtype)

    # Predict
    for idx in range(0, len(coords), batch_size):
        batch_coords = coords[idx:idx + batch_size]
        actual_bs = len(batch_coords)
        print(f"{idx+1}/{len(coords)} (step: {batch_size})", end="\r")
        for batch_idx, (y, x) in enumerate(batch_coords):
            batch[batch_idx] = image[y:y + win_size, x:x + win_size]

        # Slice to actual batch size for the (possibly) final partial batch.
        model_input = batch if actual_bs == batch_size else batch[:actual_bs]
        pred = model.predict(model_input) if use_tf else sess.run(output_names, {'input': model_input})[0]

        # Merge prediction patches immediately to avoid retaining every patch output.
        for remainder, (y, x) in enumerate(batch_coords):
            hop = pred[remainder]
            out[y:y + win_size, x:x + win_size] += hop
            mask[y:y + win_size, x:x + win_size] += 1

    out /= mask
    if manual_th is None:
        class_map = np.argmax(out, axis=-1)
    else:
        assert len(manual_th) == output_shape[-1] - 1, f"{manual_th}, {output_shape[-1]}"
        class_map = np.zeros(out.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th):
            class_map[..., idx] = np.where(out[..., idx + 1] > th, 1, 0)

    return class_map, out


# ---------------------------------------------------------------------------
# sklearn model cache (unrelated to ONNX inference)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_sklearn_model(model_name):
    return pickle.load(open(os.path.join(MODULE_PATH, f"sklearn_models/{model_name}.model"), "rb"))


def should_parallel_inference():
    """True when running both ONNX models concurrently is beneficial.

    Apple Silicon: CoreML dispatches each session to ANE independently.
    CUDA x86_64: GPU compute is independent of CPU threads.
    Pure-CPU x86_64: parallelism would over-subscribe cores -- skip.
    """
    override = os.environ.get("OEMER_PARALLEL_INFERENCE", "").strip()
    if override:
        return override == "1"
    arch = _machine_arch()
    if platform.system() == "Darwin" and arch == "arm64":
        return True
    try:
        import onnxruntime as rt
        return "CUDAExecutionProvider" in rt.get_available_providers()
    except ImportError:
        return False


def predict(region, model_name):
    if np.max(region) == 1:
        region *= 255
    m_info = _load_sklearn_model(model_name)
    model = m_info['model']
    w = m_info['w']
    h = m_info['h']
    region = Image.fromarray(region.astype(np.uint8)).resize((w, h))
    pred = model.predict(np.array(region).reshape(1, -1))
    return m_info['class_map'][pred[0]]


if __name__ == "__main__":
    img_path = "/home/kohara/omr/test_imgs/wind2.jpg"
    model_path = "./checkpoints/seg_net"
    class_map, out = inference(model_path, img_path)
