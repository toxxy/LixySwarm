"""Comparable work plans and one live connection per public requester."""
import pytest

from experiments.language_gradient_probe import requester_lock,shard_plan


def test_one_and_two_pc_plans_cover_identical_work_without_duplicate_shards():
    one,two='a'*64,'b'*64
    baseline=shard_plan([one],8,2)
    distributed=shard_plan([one,two],8)
    assert [index for index,_worker in baseline]==[index for index,_worker in distributed]==[0,1]
    assert [worker for _index,worker in baseline]==[one,one]
    assert [worker for _index,worker in distributed]==[one,two]


@pytest.mark.parametrize('workers,count,shards',[(['a'*64]*2,8,2),(['a'*64],9,2),(['a'*64,'b'*64],8,1),(['a'*64],8,17),(['bad'],8,2)])
def test_invalid_comparison_work_is_rejected(workers,count,shards):
    with pytest.raises(ValueError):shard_plan(workers,count,shards)


def test_requester_cannot_be_reused_concurrently_and_unlocks_after_failure(tmp_path):
    with pytest.raises(ValueError):
        with requester_lock('a'*64,directory=tmp_path):
            with pytest.raises(RuntimeError,match='identity is already in use'):
                with requester_lock('a'*64,timeout_s=0,directory=tmp_path):pytest.fail('duplicate requester')
            raise ValueError('fixture interrupted')
    with requester_lock('a'*64,timeout_s=0,directory=tmp_path):pass
