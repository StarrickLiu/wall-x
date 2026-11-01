"""Progress annotations: what the two passes are trained on."""

import json

import pytest

from wall_x.data.backends.lerobot.robodojo import annotations as annot

WINDOW, CANDIDATES, CAPACITY = 48, 16, 22


def write(tmp_path, record, task="swap_blocks", episode="trajectory_0"):
    directory = tmp_path / task / episode
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "progress.json").write_text(json.dumps(record), encoding="utf-8")
    return str(directory)


def annotation(tmp_path, **record):
    record.setdefault("keyframes", [])
    return annot.load(write(tmp_path, record))


# --- reading ---------------------------------------------------------------


def test_missing_file_is_not_an_error(tmp_path):
    # Those episodes train as plain action imitation.
    assert annot.load(str(tmp_path)) is None
    assert annot.AnnotationStore(str(tmp_path)).for_episode("t", "e").empty


def test_no_annotation_root_means_empty_memory():
    assert annot.AnnotationStore(None).for_episode("t", "e") is annot.EMPTY


def test_unreadable_file_degrades_to_empty_memory(tmp_path):
    directory = tmp_path / "t" / "e"
    directory.mkdir(parents=True)
    (directory / "progress.json").write_text("{not json", encoding="utf-8")
    assert annot.load(str(directory)) is None


def test_keyframes_are_sorted_and_deduplicated(tmp_path):
    a = annotation(tmp_path, keyframes=[7, 3, 7, 1])
    assert a.frames == (1, 3, 7)


def test_malformed_keyframes_degrade_to_empty_memory(tmp_path):
    assert annotation(tmp_path, keyframes=["x"]) is None


# --- boxes -----------------------------------------------------------------


def test_box_is_read_per_keyframe(tmp_path):
    a = annotation(tmp_path, keyframes=[5], meta=[{"frame": 5, "box": [10, 20, 30, 40]}])
    assert a.boxes == {5: (10, 20, 30, 40)}


@pytest.mark.parametrize("box", ([30, 20, 10, 40], [10, 40, 30, 20], [1, 2, 3], "x"))
def test_unusable_boxes_are_skipped_not_guessed(tmp_path, box):
    a = annotation(tmp_path, keyframes=[5], meta=[{"frame": 5, "box": box}])
    assert a.boxes == {}


def test_an_episode_may_mix_boxed_and_boxless_keyframes(tmp_path):
    a = annotation(tmp_path, keyframes=[5, 9],
                   meta=[{"frame": 5, "box": [1, 2, 3, 4]}, {"frame": 9}])
    assert set(a.boxes) == {5}


# --- the acceptance interval ------------------------------------------------


def test_a_window_is_the_interval_that_counts_as_correct(tmp_path):
    # What the field means is stated once, by the release's schema, rather than
    # by a version number in each file; a release that means something else is
    # refused when its index is read.
    meta = [{"frame": 5, "window": [3, 9]}]
    assert annotation(tmp_path, keyframes=[5], meta=meta).windows == {5: (3, 9)}


def test_an_entry_without_a_window_constrains_nothing(tmp_path):
    a = annotation(tmp_path, keyframes=[5], meta=[{"frame": 5}])
    assert a.windows == {}


def test_reversed_or_malformed_window_is_ignored(tmp_path):
    for window in ([9, 3], [1], "x", None):
        a = annotation(tmp_path, keyframes=[5], meta=[{"frame": 5, "window": window}])
        assert a.windows == {}


# --- memory ----------------------------------------------------------------


def test_memory_is_what_was_recorded_before_the_frame(tmp_path):
    a = annotation(tmp_path, keyframes=[0, 10, 20, 30])
    assert a.memory_before(20, capacity=22) == (0, 10)
    assert a.memory_before(0, capacity=22) == ()
    assert a.memory_before(1000, capacity=22) == (0, 10, 20, 30)


def test_memory_keeps_the_most_recent_entries(tmp_path):
    a = annotation(tmp_path, keyframes=list(range(0, 100, 10)))
    assert a.memory_before(1000, capacity=3) == (70, 80, 90)
    assert a.memory_before(1000, capacity=0) == tuple(range(0, 100, 10))


# --- selection samples -----------------------------------------------------


def test_selection_asks_about_the_candidate_nearest_each_keyframe(tmp_path):
    a = annotation(tmp_path, keyframes=[55, 97])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    assert s.candidates == (55, 61, 67, 73, 79, 85, 91, 97)
    assert s.targets == (0, 7)
    assert s.writes


def test_selection_with_no_keyframe_in_the_window_is_a_skip(tmp_path):
    a = annotation(tmp_path, keyframes=[5])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    assert s.targets == () and not s.writes
    assert s.memory == (5,)  # still recalled by the action pass


def test_a_frame_is_never_both_memory_and_candidate(tmp_path):
    a = annotation(tmp_path, keyframes=[10, 60, 90])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    assert set(s.memory).isdisjoint(s.candidates)
    assert s.memory == (10,)  # 60 and 90 are inside the window, so they are asked about


def test_slot_range_defaults_to_the_slot_itself(tmp_path):
    a = annotation(tmp_path, keyframes=[97])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    assert s.slot_ranges == ((7, 7),)


def test_acceptance_interval_widens_the_slot_range(tmp_path):
    a = annotation(tmp_path, keyframes=[97],
                   meta=[{"frame": 97, "window": [79, 97]}])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    lo, hi = s.slot_ranges[0]
    assert lo <= s.targets[0] <= hi and (lo, hi) == (4, 7)


def test_slot_range_never_excludes_its_own_slot(tmp_path):
    # An interval that does not cover the keyframe's own candidate still has to
    # accept it: that is the answer being trained.
    a = annotation(tmp_path, keyframes=[97],
                   meta=[{"frame": 97, "window": [55, 61]}])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    lo, hi = s.slot_ranges[0]
    assert lo <= s.targets[0] <= hi


def test_two_keyframes_on_one_slot_intersect_their_ranges(tmp_path):
    a = annotation(tmp_path, keyframes=[96, 97],
                   meta=[{"frame": 96, "window": [79, 97]}, {"frame": 97, "window": [85, 97]}])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    assert s.targets == (7,)
    assert s.slot_ranges == ((5, 7),)  # the tighter of the two


def test_boxes_align_with_the_targets(tmp_path):
    a = annotation(tmp_path, keyframes=[55, 97],
                   meta=[{"frame": 55, "box": [1, 2, 3, 4]}, {"frame": 97}])
    s = a.selection(100, window=48, candidates=8, capacity=22)
    assert s.targets == (0, 7)
    assert s.boxes == ((1, 2, 3, 4), None)


def test_boxes_can_be_left_out(tmp_path):
    a = annotation(tmp_path, keyframes=[55], meta=[{"frame": 55, "box": [1, 2, 3, 4]}])
    s = a.selection(100, window=48, candidates=8, capacity=22, with_boxes=False)
    assert s.boxes == (None,)


def test_selection_near_the_episode_start(tmp_path):
    a = annotation(tmp_path, keyframes=[1])
    s = a.selection(3, window=48, candidates=8, capacity=22)
    assert len(s.candidates) == 8
    assert all(0 <= c < 3 for c in s.candidates)
    assert s.writes
