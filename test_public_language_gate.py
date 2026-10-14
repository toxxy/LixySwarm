from src.learning.public_language_gate import SEED_CASES,DIALOGUE_CASES,MULTITURN,cases
from src.learning.public_conversation import public_corpus


def test_frozen_gate_matches_the_original_public_experiments_and_excludes_training_prompts():
    from experiments.evaluate_conversation_seed import CASES as seed
    from experiments.evaluate_public_dialogue import CASES as dialogue,MULTITURN as turns
    assert SEED_CASES==seed and DIALOGUE_CASES==dialogue and MULTITURN==turns
    assert len(cases())==40 and len({row[0] for row in cases()})==40
    prompts={row['prompt'] for row in public_corpus()['train']}
    assert not prompts & {messages[-1]['content'] for _,messages,_ in cases()}
