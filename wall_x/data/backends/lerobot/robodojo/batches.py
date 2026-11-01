"""An order in which samples of one kind arrive together.

An action sample carries an action chunk and a prompt with action tokens in it;
a selection sample carries neither. The model pairs the two up positionally --
the action rows it is given have to account for exactly the action-token
positions in the batch -- so a batch holding one of each has more rows than
positions, and the model refuses it rather than guessing.

This is a sampler rather than a batch sampler on purpose. A ``DataLoader``
takes consecutive runs of ``batch_size`` indices from its sampler, so handing
it an order made of same-kind blocks of that length gives same-kind batches,
and the loader keeps the fixed ``batch_size`` that the trainer reads off it to
advance its data cursor.

This is a constraint of training without packing, not a choice. Packing
concatenates samples into one sequence and counts the action-token positions
across the whole of it, so a packed step can hold both kinds at once. Without
it, a batch of mixed kinds has more action rows than positions and the model
refuses it.

The two kinds therefore meet at the optimizer step rather than inside a batch:
set ``gradient_accumulation_steps`` so that one update accumulates several
micro-batches, and both losses land in the same update. To make that happen on
every update rather than most of them, each kind is dealt across the epoch at
its own spacing, so any run of batches holds the kinds in roughly the
proportion the dataset does. With an accumulation of one, updates alternate
between the two objectives instead, which is a noisier path to the same place.

One consequence of accumulating rather than packing: each micro-batch averages
its own loss, so over an update the two objectives are weighted by their share
of the micro-batches, where a packed step weights them ``ar_loss_weight`` to
one whatever the mix. Multiplying ``ar_loss_weight`` by the ratio of action
samples to selection samples -- both printed at startup -- restores that.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np


class SameKindOrder:
    """A ``sampler``: indices ordered so each batch holds one kind.

    The kinds are spread evenly through the order rather than shuffled into it,
    which is what lets a few consecutive batches -- one optimizer step's worth
    of accumulation -- carry both objectives in the dataset's own proportion.

    Every rank builds the same batches from the same seed and keeps its own
    share of each kind, so all ranks get the same number of steps and the same
    proportions -- a rank that ran out early would hang the others at the next
    collective, and a rank holding one kind only would spend an update on a
    different objective from its peers.
    """

    def __init__(
        self,
        kinds: Sequence[str],
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside a world of {world_size}")
        self.kinds = list(kinds)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self._by_kind: Dict[str, List[int]] = {}
        for index, kind in enumerate(self.kinds):
            self._by_kind.setdefault(kind, []).append(index)
        self.batch_count = len(self.batches())

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for the next epoch; the batch count does not change."""
        self.epoch = int(epoch)

    def batches(self) -> List[List[int]]:
        """This rank's batches for the current epoch, the kinds spread out."""
        rng = np.random.default_rng([self.seed, self.epoch])
        placed: List[Tuple[float, List[int]]] = []
        for kind in sorted(self._by_kind):
            indices = np.asarray(self._by_kind[kind])
            if self.shuffle:
                indices = indices[rng.permutation(len(indices))]
            whole = len(indices) - len(indices) % self.batch_size
            # A short tail is dropped: filling it from the other kind is what
            # the model refuses, and a smaller last batch would make one step's
            # gradient a different size from every other step's.
            of_this_kind = [indices[start:start + self.batch_size].tolist()
                            for start in range(0, whole, self.batch_size)]
            # Each kind is divided between the ranks before being spread, so a
            # rank's own share holds the kinds in the same proportion, and all
            # ranks get the same number of batches of each.
            per_rank = len(of_this_kind) // self.world_size
            mine = of_this_kind[self.rank::self.world_size][:per_rank]
            # Where in this rank's epoch each of its batches belongs. Dealing
            # the kinds at their own spacing, instead of shuffling all the
            # batches together, is what keeps every few consecutive batches a
            # mix: a shuffle leaves runs of one kind, and a run as long as the
            # accumulation makes that whole update single-objective.
            for position, batch in enumerate(mine):
                placed.append(((position + 0.5) / len(mine), batch))
        placed.sort(key=lambda entry: entry[0])
        return [batch for _, batch in placed]

    def __iter__(self) -> Iterator[int]:
        for batch in self.batches():
            yield from batch

    def __len__(self) -> int:
        """How many indices this rank draws -- a whole number of batches."""
        return self.batch_count * self.batch_size


__all__ = ["SameKindOrder"]
