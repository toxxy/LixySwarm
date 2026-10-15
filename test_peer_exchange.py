"""Live discovery must survive a full address book of old visitors."""
import time
from types import SimpleNamespace

from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.peer_manager import AddressBook,PeerManager


def test_connected_providers_precede_newer_rumors_in_bounded_exchange(tmp_path):
    book=AddressBook(tmp_path/'peers.json')
    live=[{'host':'8.8.8.8','port':7338,'node_id':'a'*64},
          {'host':'1.1.1.1','port':7338,'node_id':'b'*64}]
    for peer in live:book.add(**peer)
    for index in range(120):book.add(f'9.0.0.{index+1}',20000+index,node_id=f'{index+10:064x}')
    assert not any(peer['node_id'] in {'a'*64,'b'*64} for peer in book.export(100))
    manager=PeerManager(SimpleNamespace(peers=lambda:live),address_book=book)
    peers=manager.addresses_for_exchange(100)
    assert [peer['node_id'] for peer in peers[:2]]==['a'*64,'b'*64]
    assert len(peers)==100 and len({(peer['host'],peer['port']) for peer in peers})==100
    assert manager.addresses_for_exchange(0)==[]
    assert len(manager.addresses_for_exchange(1))==1


def test_fresh_receiver_discovers_a_live_provider_through_a_crowded_seed(tmp_path):
    nodes=[]
    def node(name):
        n=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,
            advertised_host='127.0.0.1',address_book_path=tmp_path/f'{name}.json',
            target_outbound=0,allow_private=True)
        n.start();nodes.append(n);return n
    try:
        seed=node('seed');provider=node('provider')
        assert provider.connect_peer('127.0.0.1',seed.port)
        deadline=time.monotonic()+5
        while provider.identity.node_id_hex not in seed.connected_node_ids:
            assert time.monotonic()<deadline;time.sleep(.02)
        for index in range(120):
            seed.address_book.add('127.0.0.1',20000+index,node_id=f'{index+10:064x}')
        assert not any(row['node_id']==provider.identity.node_id_hex for row in seed.address_book.export(100))
        receiver=node('receiver');assert receiver.connect_peer('127.0.0.1',seed.port)
        deadline=time.monotonic()+5
        while not receiver.address_book.addresses_for_node(provider.identity.node_id_hex):
            assert time.monotonic()<deadline;time.sleep(.02)
        assert receiver.connect_node(provider.identity.node_id_hex)
        assert next(peer for peer in receiver.peers() if peer['node_id']==provider.identity.node_id_hex)['encrypted']
    finally:
        for n in reversed(nodes):n.stop()


def test_already_connected_receiver_discovers_later_provider_without_reconnecting_seed(tmp_path):
    nodes=[]
    def node(name,relay=False):
        n=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,address_book_path=tmp_path/f'{name}.json',
            target_outbound=0,allow_private=True,relay_enabled=relay)
        n.start();nodes.append(n);return n
    def wait_for(condition):
        deadline=time.monotonic()+5
        while not condition():
            assert time.monotonic()<deadline;time.sleep(.02)
    try:
        seed=node('seed',True);receiver=node('receiver')
        assert receiver.connect_peer('127.0.0.1',seed.port)
        wait_for(lambda:receiver.identity.node_id_hex in seed.connected_node_ids)
        seed_connection=next(peer['connected_at'] for peer in receiver.peers() if peer['node_id']==seed.identity.node_id_hex)
        for index in range(120):seed.address_book.add('127.0.0.1',20000+index,node_id=f'{index+10:064x}')
        provider=node('late-provider')
        assert not receiver.address_book.addresses_for_node(provider.identity.node_id_hex)
        assert provider.connect_peer('127.0.0.1',seed.port)
        wait_for(lambda:bool(receiver.address_book.addresses_for_node(provider.identity.node_id_hex)))
        original=receiver._connect_address
        async def nat_connect(host,port,**kwargs):
            return False if port==provider.port else await original(host,port,**kwargs)
        receiver._connect_address=nat_connect
        assert receiver.connect_node(provider.identity.node_id_hex)
        route=next(peer for peer in receiver.peers() if peer['node_id']==provider.identity.node_id_hex)
        assert route['transport']=='relay' and route['encrypted']
        assert next(peer['connected_at'] for peer in receiver.peers() if peer['node_id']==seed.identity.node_id_hex)==seed_connection
    finally:
        for n in reversed(nodes):n.stop()
