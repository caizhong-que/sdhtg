import torch

from sdhtg.data.shortcuts import (
    mask_entity_to_unk,
    mask_explicit_status_words,
    shuffle_entity_ids,
    status_word_mask_from_vocab,
)


def test_shuffle_entity_ids_preserves_pad_and_unk():
    vocab_size = 8
    batch = {"entity_id": torch.arange(vocab_size).view(1, -1)}
    shuffle_entity_ids(batch, entity_vocab_size=vocab_size, seed=42)
    shuffled = batch["entity_id"].flatten().tolist()
    # The permutation must be a bijection over the whole vocabulary ...
    assert sorted(shuffled) == list(range(vocab_size))
    # ... that keeps PAD (0) and UNK (1) stable.
    assert shuffled[0] == 0 and shuffled[1] == 1


def test_shuffle_entity_ids_deterministic_across_calls():
    first = {"entity_id": torch.tensor([[2, 3, 4]])}
    second = {"entity_id": torch.tensor([[2, 3, 4]])}
    shuffle_entity_ids(first, 8, seed=7)
    shuffle_entity_ids(second, 8, seed=7)
    assert torch.equal(first["entity_id"], second["entity_id"])


def test_mask_helpers_only_touch_valid_positions():
    batch = {
        "entity_id": torch.tensor([[5, 6, 7]]),
        "status_id": torch.tensor([[2, 3, 4]]),
        "mask": torch.tensor([[True, False, False]]),
    }
    mask_entity_to_unk(batch)
    assert batch["entity_id"].tolist() == [[1, 6, 7]]
    vocab = {"<PAD>": 0, "<UNK>": 1, "failed": 2, "success": 3, "timeout": 4}
    mask_explicit_status_words(batch, status_word_mask_from_vocab(vocab))
    assert batch["status_id"].tolist() == [[1, 3, 4]]
