"""Bounded lossless chunk encoding; stored artifacts and their hashes never change."""
import base64
import zlib

CHUNK_BYTES = 96 * 1024
ENCODING = 'byteplane4-zlib-v1'


def encode_chunk(chunk):
    if not isinstance(chunk, bytes) or not 0 < len(chunk) <= CHUNK_BYTES:
        raise ValueError('Invalid artifact encoding input bound')
    shuffled = b''.join(chunk[lane::4] for lane in range(4))
    compressed = zlib.compress(shuffled, level=1)
    encoding, data = (ENCODING, compressed) if len(compressed) + 128 < len(chunk) else ('raw', chunk)
    return {'encoding': encoding, 'data': base64.b64encode(data).decode('ascii')}


def decode_chunk(value, length):
    if type(length) is not int or not 0 < length <= CHUNK_BYTES:
        raise ValueError('Invalid artifact decoding output bound')
    if (not isinstance(value, dict) or not isinstance(value.get('encoding'), str)
            or value['encoding'] not in {'raw', ENCODING}):
        raise ValueError('Unknown artifact chunk encoding')
    data = value.get('data')
    if not isinstance(data, str) or len(data) > 4 * ((length + 2) // 3):
        raise ValueError('Artifact encoded chunk exceeds its bound')
    try:
        packed = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid artifact chunk base64') from exc
    if len(packed) > length:
        raise ValueError('Artifact compressed input exceeds its bound')
    if value['encoding'] == 'raw':
        chunk = packed
    else:
        decoder = zlib.decompressobj()
        try:
            shuffled = decoder.decompress(packed, length + 1)
        except zlib.error as exc:
            raise ValueError('Invalid artifact compressed stream') from exc
        if (len(shuffled) != length or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError('Artifact decompression size or stream boundary mismatch')
        chunk = bytearray(length)
        offset = 0
        for lane in range(4):
            size = max(0, (length + 3 - lane) // 4)
            chunk[lane::4] = shuffled[offset:offset + size]
            offset += size
        chunk = bytes(chunk)
    if len(chunk) != length:
        raise ValueError('Artifact decoded chunk length mismatch')
    return chunk
