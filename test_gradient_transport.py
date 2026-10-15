"""Numerical bounds and corruption handling for the unapproved research format."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from src.learning.experimental_gradient_transport import pack, unpack, pack_half, unpack_half, error, save, load, BLOCK


def test_outlier_blocks_zeros_and_tail_keep_direction_and_norm():
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(3, BLOCK + 5, generator=generator) * .001
    x[0, 0] = .8
    before=x.clone()
    original = {'a': x, 'zero': torch.zeros(2, 19)}
    decoded = unpack(pack(original), {k: t.shape for k, t in original.items()})
    metric = error(original, decoded)
    assert metric['cosine_similarity'] > .999
    assert metric['relative_l2_error'] < .03
    assert metric['decoded_norm'] <= 1.000001
    assert torch.equal(decoded['zero'], original['zero'])
    assert torch.equal(x, before)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 2.])
def test_nonfinite_and_unclipped_source_rejected(bad):
    with pytest.raises(ValueError): pack({'a': torch.tensor([bad])})


def test_unknown_keys_negative_scales_and_noncanonical_values_rejected():
    original = {'a': torch.zeros(259)}; packed = pack(original)
    with pytest.raises(ValueError): unpack({**packed, 'foreign': torch.zeros(1)}, {'a': (259,)})
    for value in [-1., float('inf')]:
        changed = {k: t.clone() for k, t in packed.items()}; changed['s.a'][0] = value
        with pytest.raises(ValueError): unpack(changed, {'a': (259,)})
    for value in [-128, 1]:
        changed = {k: t.clone() for k, t in packed.items()}; changed['q.a'][0] = value
        with pytest.raises(ValueError): unpack(changed, {'a': (259,)})
    with pytest.raises(ValueError): unpack(packed, {'a': (258,)})
    with pytest.raises(ValueError): unpack(packed, {'a': (-1,)})
    with pytest.raises(ValueError): unpack(packed, {'a': (2**30,)})


def test_source_identity_metadata_and_existing_output_are_bound(tmp_path):
    from safetensors.torch import save_file
    source = tmp_path/'source'; destination = tmp_path/'quantized'
    metadata = {'format': 'lixyswarm-language-gradient-v1', 'dataset_id': 'a'*64, 'start': '0'}
    x = {'a': torch.tensor([.01, -.02, 0])}
    save_file(x, source, metadata=metadata)
    identifier = save(source, destination)
    decoded = load(destination, source_id=identifier, metadata=metadata, shapes={'a': (3,)})
    assert error(x, decoded)['relative_l2_error'] < .01
    with pytest.raises(ValueError): load(destination, source_id='b'*64, metadata=metadata, shapes={'a': (3,)})
    with pytest.raises(ValueError): load(destination, source_id=identifier, metadata={**metadata, 'start': '1'}, shapes={'a': (3,)})
    with pytest.raises(ValueError): save(source, destination)
    assert source.read_bytes() != destination.read_bytes()


def test_half_preserves_tail_small_values_and_clipping():
    source={'a':torch.tensor([1.,.0001,1e-8,0.])*.999}
    packed=pack_half(source);decoded=unpack_half(packed,{'a':(4,)})
    assert error(source,decoded)['relative_l2_error']<.001
    assert error(source,decoded)['decoded_norm']<=1.000001
    for corrupted in [{'a':torch.ones(4)},{'a':torch.tensor([float('inf')]*4,dtype=torch.float16)}]:
        with pytest.raises(ValueError):unpack_half(corrupted,{'a':(4,)})
    with pytest.raises(ValueError):unpack_half(packed,{'a':(3,)})


def test_half_file_domain_cannot_be_used_as_block8(tmp_path):
    from safetensors.torch import save_file
    source=tmp_path/'source';destination=tmp_path/'half'
    metadata={'format':'lixyswarm-language-gradient-v1'}
    save_file({'a':torch.tensor([.1,-.2])},source,metadata=metadata)
    identifier=save(source,destination,encoding='half')
    assert error({'a':torch.tensor([.1,-.2])},load(destination,source_id=identifier,metadata=metadata,shapes={'a':(2,)},encoding='half'))['relative_l2_error']<.001
    with pytest.raises(ValueError):load(destination,source_id=identifier,metadata=metadata,shapes={'a':(2,)})


@pytest.mark.parametrize('encoding',['half','block8'])
def test_live_float32_gradient_loader_rejects_research_file(tmp_path,encoding):
    from safetensors.torch import save_file
    from src.learning.language_gradient import metadata,load_gradient
    source=tmp_path/'public';destination=tmp_path/'research'
    fields={'base_hash':'a'*64,'adapter_id':'b'*64,'dataset_id':'c'*64,'start':0,'count':2,'tokens':4}
    save_file({'a':torch.tensor([.1,-.2])},source,metadata=metadata(**fields))
    save(source,destination,encoding=encoding)
    with pytest.raises(ValueError,match='Gradient version, data or shard'):
        load_gradient(object(),destination,**fields)
