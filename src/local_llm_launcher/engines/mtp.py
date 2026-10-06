"""MTP (multi-token prediction): turn one checkbox into the flags the installed engine accepts.

Each result is {'level', 'message', 'args'}: 'red' blocks the launch, and
'args' go before any extra raw flags so a user's own flags still win.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shlex
import struct

# Before vLLM 0.11 accepted the generic "mtp", each family had its own method name.
# Families added later only ever used "mtp", so this table never needs to grow.
_LEGACY_VLLM_METHODS = {
    'deepseek_v3': 'deepseek_mtp', 'MiMoForCausalLM': 'deepseek_mtp',
    'Glm4MoeForCausalLM': 'deepseek_mtp', 'ernie4_5_moe': 'ernie_mtp',
    'qwen3_next': 'qwen3_next_mtp',
}
# config.json keys that count a model's MTP layers, across the families vLLM supports.
_LAYER_KEYS = ('num_nextn_predict_layers', 'mtp_num_hidden_layers', 'num_mtp_modules')
# An engine update is never the only way forward: the model runs fine without MTP.
_RUN_WITHOUT = ' Or turn off MTP to run the model without it.'
_VLLM_UPDATE = {
    'vllm-native': 'Update vLLM (pip install -U vllm, or Settings → Build current engine source).' + _RUN_WITHOUT,
    'vllm-docker': 'Update the image with: docker pull vllm/vllm-openai:latest.' + _RUN_WITHOUT,
}


# llama.cpp: first build whose graph runs MTP for each GGUF architecture, read from
# the source through build 11235 (_LLAMA_CHECKED_THROUGH). The --spec-type draft-mtp
# flag itself has been the same since MTP first shipped in build 9180.
_LLAMA_MTP_SINCE = {
    'qwen35': 9180, 'qwen35moe': 9180, 'step35': 9480, 'gemma4': 9549, 'cohere2moe': 9626,
    'hy_v3': 9993, 'glm-dsa': 10174, 'mimo2': 10184, 'deepseek4': 10228, 'deepseek32': 10237,
    'qwen3next': 10238, 'deepseek2': 10251, 'nemotron_h_moe': 10344, 'bailingmoe3': 10470,
    'glm4moe': 10603,
}
_LLAMA_CHECKED_THROUGH = 11235
# Gemma 4 ships its MTP head only as a separate GGUF file.
_LLAMA_SIDECAR_ONLY = {'gemma4'}
_LLAMA_UPDATE = 'Update llama.cpp (Settings → Build current engine source, or install a newer release).' + _RUN_WITHOUT
# Before build 9235 llama.cpp drafted up to 16 tokens; 3 is its tuned default since.
# The launcher defaults to 1: on 2026-10-05, build 1537a0a8 crashed or froze three
# times with 3 (two Qwen3.8 27B files, two RTX 5060 Ti) and held up with 1.
_LLAMA_DRAFT_DEFAULT, _LLAMA_DRAFT_RANGE = 1, (1, 16)


def _draft_max(config):
    """The words-ahead count as a whole number in range, or None if invalid."""
    value = config.get('mtp_draft_max', _LLAMA_DRAFT_DEFAULT)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    low, high = _LLAMA_DRAFT_RANGE
    return value if low <= value <= high else None


def _result(level, message, args=()):
    return {'level': level, 'message': message, 'args': list(args)}


def _model_config(model):
    cfg = (model or {}).get('config')
    return cfg if isinstance(cfg, dict) else {}


def _identities(cfg):
    """model_type and architecture names, including a multimodal model's text part."""
    text = cfg.get('text_config') if isinstance(cfg.get('text_config'), dict) else {}
    names = {cfg.get('model_type'), text.get('model_type')}
    for section in (cfg, text):
        names.update(a for a in section.get('architectures') or () if isinstance(a, str))
    return {name for name in names if isinstance(name, str) and name}


def _layers(cfg):
    """MTP layer count from config.json, or None when the model does not say."""
    text = cfg.get('text_config') if isinstance(cfg.get('text_config'), dict) else {}
    mtp_config = cfg.get('mtp_config') if isinstance(cfg.get('mtp_config'), dict) else {}
    for section in (cfg, text, mtp_config):
        for key in _LAYER_KEYS:
            value = section.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _family(cfg):
    return cfg.get('model_type') or next(iter(cfg.get('architectures') or ()), None) or 'this'


def vllm(mode, model, config, evidence):
    """Choose the --speculative-config for the installed vLLM, or explain why it can't."""
    if not config.get('use_mtp'):
        return _result('green', '')
    cfg = _model_config(model)
    evidence = evidence or {}
    update = _VLLM_UPDATE.get(mode, _VLLM_UPDATE['vllm-native'])
    version = f"vLLM {evidence['version']}" if evidence.get('version') else 'The installed vLLM'
    layers = _layers(cfg)
    if layers == 0:
        return _result('red', 'This model has no MTP layers (its config lists 0), so there is nothing for MTP to use. Turn MTP off.')
    flags = evidence.get('flags')
    if flags is not None and '--speculative-config' not in flags:
        return _result('red', f'{version} has no --speculative-config option, so it cannot run MTP. {update}')
    runtime = evidence.get('mtp')
    ids = _identities(cfg)
    notes = []
    models = (runtime or {}).get('models')
    # vLLM matches some families by prefix (e.g. "exaone4_5" in model_type).
    known = models is not None and any(i == name or (len(name) >= 5 and i.startswith(name))
                                       for i in ids for name in models)
    # Only a model that has MTP layers should ever be told to update the engine.
    if layers is None and not known:
        return _result('red', "This model's config does not list any MTP layers, so there is nothing for MTP to use. Turn MTP off.")
    if runtime is None:
        method = 'mtp'
        notes.append("Could not read which MTP methods this vLLM accepts; assuming vLLM 0.11 or newer, which accepts 'mtp' for every supported model.")
    else:
        methods = runtime.get('methods') or []
        if 'mtp' in methods:
            method = 'mtp'
        else:
            method = next((_LEGACY_VLLM_METHODS[i] for i in sorted(ids) if _LEGACY_VLLM_METHODS.get(i) in methods), None)
        if not methods:
            return _result('red', f'{version} cannot run MTP. {update}')
        if method is None or (models is not None and not known):
            return _result('red', f'{version} does not support MTP for {_family(cfg)} models. Newer vLLM versions support more model families. {update}')
        if models is None:
            notes.append('Could not confirm this vLLM supports MTP for this model family; vLLM checks at startup.')
    count = layers or 1
    spec = json.dumps({'method': method, 'num_speculative_tokens': count}, separators=(',', ':'))
    try:
        raw = shlex.split(str(config.get('extra_args') or ''))
    except ValueError:
        raw = []
    if any(token.split('=', 1)[0].replace('_', '-') == '--speculative-config' for token in raw):
        notes.append('Your extra raw flags also set --speculative-config; those take priority.')
    level = 'yellow' if notes else 'green'
    message = ' '.join([f"Uses the model's built-in MTP layers (vLLM method '{method}', {count} guessed token{'s' if count > 1 else ''} per step).", *notes])
    return _result(level, message, ['--speculative-config', spec])


_GGUF_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_FORMATS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


# Real headers nest arrays at most once or twice; anything deeper is corrupt.
_GGUF_MAX_DEPTH = 8
# Short top-level number arrays (per-layer settings such as head counts) are kept;
# long ones (tokenizer scores) are skipped.
_GGUF_MAX_KEPT_ARRAY = 4096


def _gguf_remaining(stream):
    return os.fstat(stream.fileno()).st_size - stream.tell()


def _gguf_read(stream, size):
    # A declared size beyond the file end is corrupt; never try to allocate it.
    if size > _gguf_remaining(stream):
        raise ValueError('truncated GGUF header')
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('truncated GGUF header')
    return data


def _gguf_string(stream):
    return _gguf_read(stream, struct.unpack('<Q', _gguf_read(stream, 8))[0]).decode('utf-8', 'replace')


def _gguf_value(stream, kind, keep, depth=0):
    if kind == 8:
        return _gguf_string(stream)
    if kind == 9:
        if depth >= _GGUF_MAX_DEPTH:
            raise ValueError('GGUF arrays nested too deeply')
        inner, count = struct.unpack('<IQ', _gguf_read(stream, 12))
        # Every element takes at least one byte (a string at least its 8-byte length).
        if count * _GGUF_SCALARS.get(inner, 8) > _gguf_remaining(stream):
            raise ValueError('truncated GGUF header')
        if inner in _GGUF_SCALARS and keep and depth == 0 and count <= _GGUF_MAX_KEPT_ARRAY:
            data = _gguf_read(stream, _GGUF_SCALARS[inner] * count)
            return struct.unpack(f'<{count}{_GGUF_FORMATS[inner][1]}', data)
        if inner in _GGUF_SCALARS:
            stream.seek(_GGUF_SCALARS[inner] * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _gguf_value(stream, inner, False, depth + 1)
        return None
    if kind not in _GGUF_FORMATS:
        raise ValueError('unknown GGUF value type')
    value = struct.unpack(_GGUF_FORMATS[kind], _gguf_read(stream, _GGUF_SCALARS[kind]))[0]
    return value if keep else None


@lru_cache(maxsize=64)
def _gguf_header(path, _stamp):
    """(metadata, tensor names) from a GGUF v2/v3 header; tensor data is never read.

    Metadata holds scalars, strings and short number arrays (as tuples)."""
    with open(path, 'rb') as stream:
        magic, version = struct.unpack('<4sI', _gguf_read(stream, 8))
        if magic != b'GGUF' or version < 2:
            raise ValueError('not a GGUF v2+ file')
        tensors, count = struct.unpack('<QQ', _gguf_read(stream, 16))
        metadata = {}
        for _ in range(count):
            key = _gguf_string(stream)
            kind = struct.unpack('<I', _gguf_read(stream, 4))[0]
            value = _gguf_value(stream, kind, True)
            if value is not None:
                metadata[key] = value
        if tensors * 24 > _gguf_remaining(stream):  # name length, dims count, type, offset
            raise ValueError('truncated GGUF header')
        names = set()
        for _ in range(tensors):
            names.add(_gguf_string(stream))
            dims = struct.unpack('<I', _gguf_read(stream, 4))[0]
            stream.seek(8 * dims + 12, os.SEEK_CUR)  # dims, type, offset
    return metadata, frozenset(names)


def _gguf(path):
    """Header of a GGUF, joining the tensor names of every split of a sharded model."""
    stat = os.stat(path)
    metadata, names = _gguf_header(path, (stat.st_mtime_ns, stat.st_size))
    split = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', path)
    if split:
        names = set(names)
        for index in range(1, int(split[2]) + 1):
            part = path[:split.start()] + f'-{index:05d}-of-{split[2]}.gguf'
            if part != path and os.path.exists(part):
                part_stat = os.stat(part)
                names |= _gguf_header(part, (part_stat.st_mtime_ns, part_stat.st_size))[1]
    return metadata, names


def _gguf_mtp_layers(path):
    """(architecture, whether the file holds MTP layers llama.cpp can load)."""
    metadata, names = _gguf(path)
    arch = metadata.get('general.architecture')
    blocks = metadata.get(f'{arch}.block_count')
    layers = metadata.get(f'{arch}.nextn_predict_layers') or 0
    # llama.cpp's own check: the last block carries the NextN projection.
    has = bool(layers) and isinstance(blocks, int) and f'blk.{blocks - 1}.nextn.eh_proj.weight' in names
    return arch, has


def is_head(name):
    """llama.cpp's rule for a separate MTP head file: a GGUF whose name contains 'mtp-'."""
    return 'mtp-' in name.lower() and name.lower().endswith('.gguf')


def _head_arch(path):
    try:
        return _gguf(path)[0].get('general.architecture')
    except (OSError, ValueError, struct.error, UnicodeError, MemoryError, RecursionError, OverflowError):
        return None


def _sidecar(model, model_path, arch):
    """A separate MTP head for this model: same folder first, then same quant.

    A head must be built for the model's architecture: llama.cpp's converter keeps it
    ('qwen35'), and Gemma 4 names its head '<arch>-assistant'. Any other mtp- file in a
    shared folder belongs to a different model.
    """
    from ..discovery import guess_gguf_quant
    folder = Path(model_path).parent
    heads = [f['path'] for f in model.get('gguf_files') or () if is_head(f.get('filename', ''))]
    try:
        heads += [str(p) for p in folder.iterdir() if is_head(p.name)]
    except OSError:
        pass
    heads = sorted(h for h in set(heads) - {model_path}
                   if arch and (_head_arch(h) == arch or str(_head_arch(h) or '').startswith(arch + '-')))
    if not heads:
        return None
    quant = (guess_gguf_quant(Path(model_path).name) or '').lower()
    return min(heads, key=lambda h: (Path(h).parent != folder, not quant or quant not in Path(h).name.lower(), h))


def llamacpp(model, config, capabilities, path=None):
    """Choose --spec-type draft-mtp flags for the installed llama-server, or explain why it can't."""
    if not config.get('use_mtp'):
        return _result('green', '')
    draft_max = _draft_max(config)
    if draft_max is None:
        low, high = _LLAMA_DRAFT_RANGE
        return _result('red', f'Set "MTP words drafted ahead" to a whole number from {low} to {high}.')
    if model.get('format') != 'gguf' and not model.get('gguf_files'):
        return _result('red', 'llama.cpp runs GGUF files; this model has none. Turn MTP off.')
    if path is None:
        from .llamacpp import pick_gguf_path
        path = pick_gguf_path(model, config)
    try:
        arch, embedded = _gguf_mtp_layers(path)
    except (OSError, ValueError, struct.error, UnicodeError, MemoryError, RecursionError, OverflowError):
        return _result('red', 'Could not read the selected GGUF file to check for MTP layers. Turn MTP off.')
    head = None if embedded and arch not in _LLAMA_SIDECAR_ONLY else _sidecar(model, path, arch)
    if not embedded and head is None:
        extra = (' Gemma 4 keeps its MTP head in a separate file; download it next to the model (its name contains "mtp-").'
                 if arch in _LLAMA_SIDECAR_ONLY else '')
        return _result('red', f'This GGUF file has no MTP layers and there is no separate MTP head file for this model next to it, so there is nothing for MTP to use.{extra} Turn MTP off.')
    # Only a file that has MTP layers should ever be told to update the engine.
    if capabilities.get('mtp') is False:
        build = capabilities.get('build')
        where = f'This llama.cpp (build {build})' if build else 'This llama.cpp'
        return _result('red', f'{where} has no MTP support (it arrived in build 9180). {_LLAMA_UPDATE}')
    notes = []
    build = capabilities.get('build')
    since = _LLAMA_MTP_SINCE.get(arch)
    if since is None:
        if not isinstance(build, int) or build <= _LLAMA_CHECKED_THROUGH:
            return _result('red', f"llama.cpp can't use MTP with {arch or 'this'} models yet (checked through build {_LLAMA_CHECKED_THROUGH}); the layers are in the file but llama.cpp has no MTP support for this architecture. A newer build may add it. {_LLAMA_UPDATE}")
        notes.append(f'MTP support for {arch} is unconfirmed; llama.cpp checks at startup.')
    elif isinstance(build, int):
        if build < since:
            return _result('red', f'This llama.cpp (build {build}) predates MTP support for {arch} models, which arrived in build {since}. {_LLAMA_UPDATE}')
    elif capabilities.get('mtp') is None:
        notes.append(f'Could not check the installed llama-server; MTP for {arch} needs build {since} or newer.')
    else:
        notes.append(f'Could not read the llama.cpp build number; MTP for {arch} needs build {since} or newer.')
    try:
        raw = {token.split('=', 1)[0] for token in shlex.split(str(config.get('extra_args') or ''))}
    except ValueError:
        raw = set()
    # llama.cpp adds up repeated --spec-type values instead of keeping the last one,
    # so a user's own --spec-type list replaces the launcher's rather than joining it.
    args = [] if '--spec-type' in raw else ['--spec-type', 'draft-mtp']
    if '--spec-type' in raw:
        notes.append('Your extra raw flags set --spec-type, so the launcher leaves out its own; include draft-mtp there to keep MTP.')
    args += ['--spec-draft-n-max', str(draft_max)]
    if head:
        args += ['--spec-draft-model', head]
    if raw & {'--spec-draft-model', '-md', '--model-draft', '--spec-draft-n-max'}:
        notes.append('Your extra raw flags also set draft options; those take priority.')
    source = f'the separate head file {Path(head).name}' if head else "the model's built-in MTP layers"
    message = ' '.join([f'Uses {source} (--spec-type draft-mtp, up to {draft_max} guessed tokens per step).', *notes])
    return _result('yellow' if notes else 'green', message, args)


def resolve(mode, model, config, *, vllm_binary='vllm', llama_binary=None, wait=True):
    """The one MTP decision for advice (wait=False, never blocks) and launch."""
    if not config.get('use_mtp'):
        return _result('green', '')
    if mode.startswith('vllm'):
        from . import vllm_capabilities
        return vllm(mode, model, config, vllm_capabilities.probe(mode, vllm_binary, wait=wait))
    from .. import hardware
    capabilities = hardware.llama_capabilities(llama_binary) if llama_binary else {}
    return llamacpp(model, config, capabilities)
