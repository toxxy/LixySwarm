"""Research-only block int8/float16 gradients; live work accepts neither format.

The public source stays immutable. A decoded copy is an unsigned approximation,
not the artifact bound by the original worker receipt. Promotion and integration
require a separately versioned training policy and physical evaluation.
"""
import hashlib
import json
import math
from pathlib import Path

FORMAT = 'lixyswarm-research-gradient-block8-v1'
HALF_FORMAT = 'lixyswarm-research-gradient-half-v1'
BLOCK = 256
MAX_BYTES = 32 * 1024**2


def norm(tensors):
    return math.sqrt(sum(float(t.double().square().sum()) for t in tensors.values()))


def validate_shapes(shapes):
    if (not isinstance(shapes,dict) or not shapes
            or any(not isinstance(name,str) or not name or len(name)>512
                or not isinstance(shape,(tuple,list)) or not 1<=len(shape)<=4
                or any(type(n) is not int or n<=0 for n in shape) for name,shape in shapes.items())
            or sum(math.prod(shape)*4 for shape in shapes.values())>MAX_BYTES):
        raise ValueError('Research adapter shapes exceed their bounds')


def validate_source(tensors):
    import torch
    if (not tensors or any(not isinstance(k, str) or not k or len(k) > 512 for k in tensors)
            or sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_BYTES):
        raise ValueError('Research gradient exceeds tensor bounds')
    if any(t.dtype != torch.float32 or t.device.type != 'cpu' or not torch.isfinite(t).all()
           or not 1 <= t.ndim <= 4 or t.numel() == 0 for t in tensors.values()):
        raise ValueError('Research gradient must be finite CPU float32')
    if norm(tensors) > 1.00001:
        raise ValueError('Research source exceeds its clipping bound')


def pack(tensors):
    import torch
    validate_source(tensors)
    result = {}
    for name, tensor in tensors.items():
        flat = tensor.flatten(); size = flat.numel()
        padded = torch.nn.functional.pad(flat, (0, (-size) % BLOCK)).reshape(-1, BLOCK)
        scales = padded.abs().amax(dim=1) / 127
        # A zero block has exact zeros and scale zero; no division by zero.
        quantized = torch.round(padded / torch.where(scales > 0, scales, 1).unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        result['q.' + name] = quantized.flatten()[:size].reshape(tensor.shape).contiguous()
        result['s.' + name] = scales.contiguous()
    return result


def pack_half(tensors):
    import torch
    validate_source(tensors)
    return {name:t.to(torch.float16).contiguous() for name,t in tensors.items()}


def unpack_half(packed, shapes):
    import torch
    validate_shapes(shapes)
    if set(packed) != set(shapes) or not shapes:
        raise ValueError('Research half tensor names differ from the known adapter')
    if any(t.dtype != torch.float16 or tuple(t.shape) != tuple(shapes[name])
           or t.device.type != 'cpu' or not torch.isfinite(t).all() for name,t in packed.items()):
        raise ValueError('Research half tensor shape, type or values are invalid')
    result={name:t.float() for name,t in packed.items()}
    length=norm(result)
    if sum(t.numel()*t.element_size() for t in result.values())>MAX_BYTES or not math.isfinite(length) or length>1.005:
        raise ValueError('Research half reconstruction exceeds its bounds')
    return {name:t/length for name,t in result.items()} if length>1 else result


def unpack(packed, shapes):
    import torch
    validate_shapes(shapes)
    expected = {prefix + name for name in shapes for prefix in ['q.', 's.']}
    if set(packed) != expected or not shapes:
        raise ValueError('Research tensor names differ from the known adapter')
    result = {}
    for name, shape in shapes.items():
        count = math.prod(shape); q = packed['q.' + name]; scales = packed['s.' + name]
        if (q.dtype != torch.int8 or tuple(q.shape) != tuple(shape)
                or scales.dtype != torch.float32 or tuple(scales.shape) != ((count + BLOCK - 1) // BLOCK,)
                or q.device.type != 'cpu' or scales.device.type != 'cpu'
                or not torch.isfinite(scales).all() or (scales < 0).any() or (q == -128).any()):
            raise ValueError('Research quantization shape, type or scale is invalid')
        padded = torch.nn.functional.pad(q.flatten(), (0, (-count) % BLOCK)).reshape(-1, BLOCK)
        if ((scales == 0) & (padded != 0).any(dim=1)).any():
            raise ValueError('Noncanonical zero block')
        decoded = (padded.float() * scales.unsqueeze(1)).flatten()[:count].reshape(shape)
        if not torch.isfinite(decoded).all():
            raise ValueError('Nonfinite research reconstruction')
        result[name] = decoded
    size = sum(t.numel() * t.element_size() for t in result.values())
    length = norm(result)
    if size > MAX_BYTES or not math.isfinite(length) or length > 1.05:
        raise ValueError('Research reconstruction exceeds its bounds')
    # Clipping remains a hard invariant, including quantization round-off.
    if length > 1:
        result = {k: t / length for k, t in result.items()}
    return result


def save(source, destination, *, encoding='block8'):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BYTES:
        raise ValueError('Use a bounded regular public gradient')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Research output must be new')
    with safe_open(source, framework='pt', device='cpu') as opened:
        metadata = opened.metadata()
    if not metadata or metadata.get('format') != 'lixyswarm-language-gradient-v1':
        raise ValueError('Research input is not the public gradient format')
    with source.open('rb') as stream:identifier = hashlib.file_digest(stream, 'sha256').hexdigest()
    tensors = load_file(source, device='cpu')
    if encoding not in {'block8','half'}:raise ValueError('Unknown research encoding')
    packed=pack(tensors) if encoding=='block8' else pack_half(tensors)
    header={'format':FORMAT if encoding=='block8' else HALF_FORMAT,'source_id':identifier,
        'source_metadata':json.dumps(metadata,sort_keys=True,separators=(',', ':'))}
    if encoding=='block8':header['block_size']=str(BLOCK)
    save_file(packed, destination, metadata=header)
    return identifier


def load(path, *, source_id, metadata, shapes, encoding='block8'):
    from safetensors import safe_open
    from safetensors.torch import load_file
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Research encoding exceeds its file bounds')
    with safe_open(path, framework='pt', device='cpu') as opened:
        actual = opened.metadata()
    if encoding not in {'block8','half'}:raise ValueError('Unknown research encoding')
    expected = {'format': FORMAT if encoding=='block8' else HALF_FORMAT, 'source_id': source_id,
        'source_metadata': json.dumps(metadata, sort_keys=True, separators=(',', ':'))}
    if encoding=='block8':expected['block_size']=str(BLOCK)
    if actual != expected:
        raise ValueError('Research source, format or shard differs')
    return (unpack if encoding=='block8' else unpack_half)(load_file(path, device='cpu'), shapes)


def error(source, decoded):
    if set(source) != set(decoded):
        raise ValueError('Research comparison requires identical tensor names')
    before = norm(source); difference = norm({k: source[k] - decoded[k] for k in source})
    after = norm(decoded)
    dot = sum(float((source[k].double() * decoded[k].double()).sum()) for k in source)
    return {'source_norm': before, 'decoded_norm': after, 'absolute_l2_error': difference,
        'relative_l2_error': difference / before if before else 0,
        'cosine_similarity': dot / (before * after) if before and after else 1 if not before and not after else 0,
        'max_absolute_error': max(float((source[k] - decoded[k]).abs().max()) for k in source)}
