import pytest
from local_llm_launcher import hardware, advisor
from local_llm_launcher.engines import llamacpp, vllm_native, vllm_docker
from tests.test_engines import GGUF, MODEL
from tests.test_advisor import DUAL_5060TI

@pytest.mark.parametrize('config', [
    {'tensor_split': '1,0'}, {'tensor_split': 'nan,1'}, {'tensor_split': '1,'},
    {'device': 'CUDA0,CUDA1,CUDA2', 'tensor_split': '1,1'},
    {'split_mode': 'none', 'tensor_split': '1,1'},
    {'split_mode': 'tensor', 'flash_attn': 'off'},
    {'cpu_moe': True, 'n_cpu_moe': 2}, {'n_cpu_moe': -1},
])
def test_invalid_placement_rejected(config):
    with pytest.raises(ValueError): llamacpp.build(GGUF, config)
    with pytest.raises(ValueError): advisor.advise('llamacpp', GGUF, config, DUAL_5060TI)

def test_explicit_placement():
    cfg = dict(device='CUDA0,CUDA1,CUDA2', tensor_split='40,40,40', split_mode='layer', n_cpu_moe=26, ubatch_size=256, numa='distribute')
    argv = llamacpp.build(GGUF, cfg)['argv']
    for flag, value in [('--device', cfg['device']), ('--tensor-split', '40,40,40'), ('--n-cpu-moe', '26'), ('--ubatch-size', '256'), ('--numa', 'distribute')]:
        assert argv[argv.index(flag)+1] == value
    report = advisor.advise('llamacpp', GGUF, cfg, DUAL_5060TI)
    assert report['overall']['level'] == 'yellow'
    assert report['budget']['fit_unknown'] is True

@pytest.mark.parametrize('no_mmap,mlock,mode', [(False,False,None),(True,False,'none'),(False,True,'mmap+mlock'),(True,True,'mlock')])
def test_load_modes(monkeypatch, no_mmap, mlock, mode):
    monkeypatch.setattr(hardware, 'llama_capabilities', lambda binary: {'load_mode': True, 'devices': []})
    argv = llamacpp.build(GGUF, dict(no_mmap=no_mmap, mlock=mlock))['argv']
    assert '--no-mmap' not in argv and '--mlock' not in argv
    if mode: assert argv[argv.index('--load-mode')+1] == mode
    else: assert '--load-mode' not in argv
    monkeypatch.setattr(hardware, 'llama_capabilities', lambda binary: {'load_mode': False, 'devices': []})
    argv = llamacpp.build(GGUF, dict(no_mmap=no_mmap, mlock=mlock))['argv']
    assert ('--no-mmap' in argv) == no_mmap
    assert ('--mlock' in argv) == mlock

def test_interleave(monkeypatch):
    monkeypatch.setattr(hardware.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(hardware.shutil, 'which', lambda name: '/usr/bin/numactl' if name == 'numactl' else None)
    for build, model in [(llamacpp.build,GGUF),(vllm_native.build,MODEL)]:
        assert build(model, {'numactl_interleave': True})['argv'][:3] == ['/usr/bin/numactl','--interleave=all','--']
    with pytest.raises(ValueError): vllm_docker.build(MODEL, {'numactl_interleave': True})
    monkeypatch.setattr(hardware.shutil, 'which', lambda name: None)
    with pytest.raises(ValueError): vllm_native.build(MODEL, {'numactl_interleave': True})

@pytest.mark.parametrize('online,allowed,expected', [('0-1','0-1',[0,1]),('0-1','1',[1]),('0','0',[0]),('0','1',[])])
def test_numa_allowed_mask(tmp_path,monkeypatch,online,allowed,expected):
    monkeypatch.setattr(hardware.platform, 'system', lambda: 'Linux')
    nodes=tmp_path/'online'; nodes.write_text(online)
    status=tmp_path/'status'; status.write_text('Mems_allowed_list:\t'+allowed+'\n')
    assert hardware.detect_numa(nodes,status)['nodes'] == expected
    assert hardware.detect_numa(tmp_path/'missing',status)['nodes'] is None


def test_capability_probe_cached_by_binary_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    binary=tmp_path/'llama-server'; binary.write_text('binary')
    calls=[]
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='--load-mode MODE' if argv[-1]=='--help' else 'Available devices:\n  Vulkan0: Card A\n  CUDA2: Card B', stderr='')
    monkeypatch.setattr(hardware.subprocess,'run',run)
    hardware._llama_capabilities_cached.cache_clear()
    assert hardware.llama_capabilities(str(binary))['devices'][0]['name'] == 'Vulkan0'
    assert hardware.llama_capabilities(str(binary))['load_mode'] is True
    assert len(calls)==3  # --help, --list-devices, --version
    binary.write_text('new binary')
    hardware.llama_capabilities(str(binary))
    assert len(calls)==6


def test_placement_api_rejects_before_launch(monkeypatch):
    from fastapi.testclient import TestClient
    from local_llm_launcher.app import create_app
    from local_llm_launcher import api
    monkeypatch.setattr(api,'find_model',lambda _:GGUF)
    monkeypatch.setattr(api,'get_hardware',lambda:DUAL_5060TI)
    client=TestClient(create_app(),base_url='http://127.0.0.1')
    for path, fields in [('/api/advise',{'engine':'llamacpp'}),('/api/servers',{'engine_mode':'llamacpp'})]:
        response=client.post(path,json={**fields,'repo_id':GGUF['repo_id'],'config':{'split_mode':'tensor','flash_attn':'off'}})
        assert response.status_code==400
        assert 'flash attention' in response.json()['detail']
    response=client.post('/api/advise',json={'engine':'vllm','engine_mode':'vllm-docker','repo_id':'x','config':{'numactl_interleave':True}})
    assert response.status_code==400 and 'Docker' in response.json()['detail']


@pytest.mark.parametrize('count',[None,1,2])
def test_numa_advice_without_automatic_policy(count):
    hw={**DUAL_5060TI,'numa':{'node_count':count}}
    report=advisor.advise('llamacpp',GGUF,{'numa':'distribute'},hw)
    message=report['flags']['numa']['message']
    assert ('nothing to spread across' in message)==(count==1)
    argv=llamacpp.build(GGUF,{})['argv']
    assert '--numa' not in argv and '--interleave=all' not in argv


def test_raw_flags_unknown_fit_keep_order():
    report=advisor.advise('llamacpp',GGUF,{'extra_args':'--ctx-size 100000'},DUAL_5060TI)
    assert report['overall']['level']=='yellow' and report['budget']['pct'] is None
    argv=llamacpp.build(GGUF,{'ctx_size':8192,'extra_args':'--ctx-size 100000'})['argv']
    assert argv[-2:]==['--ctx-size','100000']


def test_null_offload_is_unset():
    assert advisor.advise('llamacpp', GGUF, {'n_cpu_moe': None}, DUAL_5060TI)['overall'] == advisor.advise('llamacpp', GGUF, {}, DUAL_5060TI)['overall']


@pytest.mark.parametrize('raw, expected', [
    ('0,1', '0,1'), (' 0 , 1 ', '0,1'), ('1,', '1'), (0, '0'), ('GPU-abc,1', 'GPU-abc,1'), ('01', '1'),
    ('1,0', '1,0'), ('MIG-GPU-abc/1/0', 'MIG-GPU-abc/1/0'),
    (None, None), ('', None), ('  ', None),
])
def test_normalize_device_ids(raw, expected):
    from local_llm_launcher.engines.placement import normalize_device_ids
    assert normalize_device_ids(raw) == expected


@pytest.mark.parametrize('raw', ['0 1', ',', '0,0', True, 'gpu1', '-1', '1;2', '1.0', '0,00'])
def test_malformed_device_ids_rejected_before_launch(raw):
    from local_llm_launcher.engines.placement import normalize_device_ids, validate
    with pytest.raises(ValueError):
        normalize_device_ids(raw)
    for engine in ('vllm-native', 'vllm-docker'):
        with pytest.raises(ValueError, match='GPU numbers'):
            validate(engine, {'device_ids': raw})


@pytest.mark.parametrize('raw, expected', [('0, 1', [0, 1]), ('1,0', [1, 0]), (0, [0]), (None, None)])
def test_parse_device_ids(raw, expected):
    from local_llm_launcher.engines.placement import parse_device_ids
    assert parse_device_ids(raw) == expected


def test_parse_device_ids_rejects_non_numbers():
    from local_llm_launcher.engines.placement import parse_device_ids
    with pytest.raises(ValueError):
        parse_device_ids('GPU-uuid')
