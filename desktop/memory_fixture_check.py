"""Check a known synthetic encrypted fixture in a fresh native process.

No arbitrary questions or memory contents are returned by this test mode.
The guard rejects personal stores containing anything outside public fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def check(directory, name):
    from desktop.chat_history import ChatHistory
    from src.learning.public_memory_gate import memory_cases, statement_messages, distractor_messages
    case = next(row for row in memory_cases() if row['case'] == name)
    history = ChatHistory(directory)
    allowed_keys = {'chat', 'personal_context', 'personal_context_enabled'}
    allowed_messages = statement_messages(case) + distractor_messages()
    if (not set(history.value).issubset(allowed_keys) or history.messages != distractor_messages()[-200:]
            or not history.memory_enabled or not history.share_context
            or any(message not in allowed_messages for episode in history.memory.snapshot()['episodes'] for message in episode['messages'])):
        raise ValueError('This mode only checks its exact public synthetic fixture')
    messages, recalled = history.context(history.messages, case['question'])
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if not recalled or case['expected'].encode() not in encoded:
        raise ValueError('Fresh process did not recover its public fixture')
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise ValueError('Personal memory loaded tensor models')
    return {'success': True, 'case': name, 'restored_in_fresh_process': True,
            'recalled_episodes': len(recalled), 'context_sha256': hashlib.sha256(encoded).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-fixture-check', action='store_true')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    try:
        result = check(args.directory, args.case)
    except Exception:
        print(json.dumps({'success': False, 'error': 'Synthetic memory verification failed'}))
        return 1
    print(json.dumps(result))
    return 0
