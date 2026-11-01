"""An order in which samples of one kind arrive together."""

import pytest

from wall_x.data.backends.lerobot.robodojo.batches import SameKindOrder

KINDS = ["action"] * 10 + ["selection"] * 7


def chunks(order):
    """The batches a DataLoader would take from this order."""
    indices = list(order)
    size = order.batch_size
    return [indices[i:i + size] for i in range(0, len(indices), size)]


def test_no_batch_mixes_the_two_kinds():
    # The model pairs action rows with action-token positions, so a mixed
    # batch has more rows than positions and is refused.
    for batch in chunks(SameKindOrder(KINDS, batch_size=3)):
        assert len({KINDS[i] for i in batch}) == 1


RUN_KINDS = ["action"] * 106 + ["selection"] * 60   # the proportions of a run
ACCUMULATION = 4                                    # what the configs ship


def windows(order, size=ACCUMULATION):
    """The batches of one optimizer step's worth of accumulation."""
    batches = chunks(order)
    return [batches[i:i + size] for i in range(0, len(batches) - size + 1, size)]


def kinds_in(window, kinds):
    return [kinds[index] for batch in window for index in batch]


def test_every_update_sees_both_kinds():
    """Batches are single-kind; an optimizer step is not.

    The two objectives meet at the optimizer step, which needs the kinds dealt
    through the order at their own spacing. Shuffling all the batches together
    instead leaves runs of one kind, and a run as long as the accumulation is
    an update that trains one objective only.
    """
    order = SameKindOrder(RUN_KINDS, batch_size=1)
    for window in windows(order):
        assert set(kinds_in(window, RUN_KINDS)) == {"action", "selection"}


def test_an_update_holds_the_kinds_in_the_proportion_the_data_does():
    # Accumulation weights each objective by its share of the micro-batches,
    # so that share has to be the dataset's own and not drift step to step.
    order = SameKindOrder(RUN_KINDS, batch_size=1)
    share = RUN_KINDS.count("selection") / len(RUN_KINDS)
    counts = [kinds_in(w, RUN_KINDS).count("selection") for w in windows(order)]
    assert min(counts) >= 1
    assert max(counts) - min(counts) <= 1
    assert abs(sum(counts) / len(counts) - share * ACCUMULATION) < 0.1


def test_each_rank_gets_its_own_mix_of_both():
    # The ranks take every world_size-th batch of one spread-out order, so the
    # spread has to survive that slicing: a rank seeing one kind only would
    # train a different objective from its peers on the same update.
    world = 4
    for rank in range(world):
        order = SameKindOrder(RUN_KINDS, batch_size=1, rank=rank, world_size=world)
        for window in windows(order):
            assert set(kinds_in(window, RUN_KINDS)) == {"action", "selection"}


def test_the_order_is_a_whole_number_of_batches():
    order = SameKindOrder(KINDS, batch_size=3)
    assert len(order) % order.batch_size == 0
    assert all(len(batch) == 3 for batch in chunks(order))


def test_a_short_tail_is_dropped():
    # 10 action + 7 selection at 3 each: three full action batches, two
    # selection, and five samples left over.
    assert SameKindOrder(KINDS, batch_size=3).batch_count == 5


def test_a_batch_of_one_keeps_everything():
    assert SameKindOrder(KINDS, batch_size=1).batch_count == len(KINDS)


def test_every_index_appears_once():
    order = list(SameKindOrder(KINDS, batch_size=1))
    assert sorted(order) == list(range(len(KINDS)))


def test_every_rank_takes_the_same_number_of_steps():
    # A rank that ran out first would hang the others at the next collective.
    counts = {SameKindOrder(KINDS, batch_size=2, rank=r, world_size=3).batch_count
              for r in range(3)}
    assert len(counts) == 1


def test_the_ranks_divide_the_samples_between_them():
    world = 3
    seen = [list(SameKindOrder(KINDS, batch_size=2, rank=r, world_size=world))
            for r in range(world)]
    flat = [index for rank in seen for index in rank]
    assert len(flat) == len(set(flat))


def test_an_epoch_reshuffles_without_changing_the_step_count():
    order = SameKindOrder(KINDS, batch_size=2)
    first = list(order)
    order.set_epoch(1)
    second = list(order)
    assert first != second
    assert len(first) == len(second)


def test_the_same_seed_and_epoch_give_the_same_order():
    assert (list(SameKindOrder(KINDS, batch_size=2, seed=7))
            == list(SameKindOrder(KINDS, batch_size=2, seed=7)))


def test_another_seed_gives_another_order():
    assert (list(SameKindOrder(KINDS, batch_size=2, seed=7))
            != list(SameKindOrder(KINDS, batch_size=2, seed=8)))


def test_shuffling_off_keeps_the_samples_in_order():
    assert chunks(SameKindOrder(KINDS, batch_size=2, shuffle=False))[0] == [0, 1]


def test_one_kind_is_fine():
    assert SameKindOrder(["action"] * 8, batch_size=4).batch_count == 2


def test_a_batch_bigger_than_either_kind_yields_nothing():
    # Better than a batch that silently mixes kinds to fill itself.
    assert SameKindOrder(KINDS, batch_size=20).batch_count == 0


def test_an_impossible_batch_size_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        SameKindOrder(KINDS, batch_size=0)


def test_a_rank_outside_the_world_is_refused():
    with pytest.raises(ValueError, match="outside a world"):
        SameKindOrder(KINDS, batch_size=2, rank=4, world_size=4)
