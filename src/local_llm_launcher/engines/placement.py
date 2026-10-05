"""Validation shared by advice and command builders; native memory policy wrapper."""
import math
import re
from .. import hardware


_GPU_ITEM = re.compile(r'[0-9]+|(?:GPU|MIG)-[0-9A-Za-z/-]+')  # nvidia-smi number or CUDA UUID


def normalize_device_ids(raw):
    """The one spelling of a GPU list that advice, checks and the engine all use.

    Returns None when unset. Raises ValueError for anything but a comma list of
    distinct nvidia-smi numbers or CUDA GPU UUIDs (so not '0 1', 'gpu1' or '0,00').
    """
    text = '' if raw is None else str(raw).strip()
    if not text:
        return None
    items = [item.strip() for item in text.split(',') if item.strip()]
    items = [str(int(item)) if item.isascii() and item.isdigit() else item for item in items]
    if (isinstance(raw, bool) or not items or len(set(items)) != len(items)
            or not all(_GPU_ITEM.fullmatch(item) for item in items)):
        raise ValueError('Enter GPU numbers separated by commas, such as 0,1.')
    return ','.join(items)


def parse_device_ids(raw):
    """GPU numbers (nvidia-smi order) in the order given; None when unset. Raises ValueError."""
    ids = normalize_device_ids(raw)
    return None if ids is None else [int(item) for item in ids.split(',')]


def validate(engine, config, numa=None):
    if config.get('numactl_interleave'):
        if engine == 'vllm-docker':
            raise ValueError('Memory interleaving supports native engines only; turn it off for Docker.')
        numa = numa if numa is not None else hardware.detect_numa()
        if not numa.get('linux') or not numa.get('numactl_path'):
            raise ValueError('Memory interleaving requires Linux and numactl installed on PATH.')
    if engine.startswith('vllm'):
        normalize_device_ids(config.get('device_ids'))
    if engine != 'llamacpp':
        return
    mode = config.get('split_mode') or 'layer'
    if mode not in ('layer', 'row', 'none', 'tensor'):
        raise ValueError('Choose layer, row, none, or tensor for the GPU split style.')
    devices = str(config.get('device') or '').split(',') if config.get('device') else []
    if devices and (any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', d) for d in devices) or len(set(devices)) != len(devices)):
        raise ValueError('Use distinct llama.cpp device names separated by commas, such as CUDA0,CUDA1.')
    if 'none' in devices and len(devices) != 1:
        raise ValueError('Device none cannot be combined with GPU devices.')
    ratios = config.get('tensor_split')
    if ratios is not None:
        try:
            values = [float(v) for v in str(ratios).split(',')]
            if not values or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError()
        except ValueError:
            raise ValueError('GPU split proportions must be positive finite numbers, such as 40,40,40.') from None
        if mode == 'none' or devices == ['none']:
            raise ValueError('GPU proportions require a multi-GPU split style and GPU devices.')
        if devices and len(values) != len(devices):
            raise ValueError('Provide one GPU split proportion per selected device.')
    for key in ('main_gpu', 'n_cpu_moe', 'ubatch_size'):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == 'ubatch_size' else 0)):
            raise ValueError(f'{key} must be a whole number at least {1 if key == "ubatch_size" else 0}.')
    if devices and config.get('main_gpu') is not None and config['main_gpu'] >= len(devices):
        raise ValueError('Main GPU index must refer to a selected device (numbered from zero).')
    if config.get('cpu_moe') and config.get('n_cpu_moe', 0):
        raise ValueError('Choose all CPU experts or a CPU expert layer count, not both.')
    if config.get('numa') not in (None, 'distribute', 'isolate', 'numactl'):
        raise ValueError('NUMA mode must be distribute, isolate, numactl, or unset.')
    if mode == 'tensor':
        if config.get('flash_attn') == 'off':
            raise ValueError('Tensor splitting requires flash attention on or auto.')
        # Upstream PR #23792 added tensor splitting with quantized KV cache.
        # Model/backend-specific support belongs to the installed engine, not
        # a blanket ban based on cache precision or model weight quantization.


def wrap(argv, config):
    if config.get('numactl_interleave'):
        return [hardware.detect_numa()['numactl_path'], '--interleave=all', '--'] + argv
    return argv
