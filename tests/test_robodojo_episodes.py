"""Binding annotations to the episodes they describe."""

import json

import pytest

from wall_x.data.backends.lerobot.robodojo.episodes import EpisodeIndex, Task

COVER = "Cover the blocks, remember their colors, then uncover them."
SWAP = "Swap the two blocks using the empty mat."
PLAIN = "Sort the objects by category into the three baskets."

TASKS = (Task("cover_blocks", COVER, 3), Task("swap_blocks", SWAP, 2))
# Two annotated tasks in contiguous blocks, with an unannotated task between.
EPISODES = [
    (0, PLAIN, 500), (1, PLAIN, 501),
    (2, COVER, 540), (3, COVER, 541), (4, COVER, 542),
    (5, SWAP, 430), (6, SWAP, 431),
]


def index(root=None, tasks=TASKS, episodes=EPISODES):
    return EpisodeIndex(tasks=tasks, episodes=episodes, root=root)


def test_an_episode_binds_to_its_position_in_the_task_block():
    found = index()
    assert found.binding(2).episode == "trajectory_0"
    assert found.binding(4).episode == "trajectory_2"
    assert found.binding(5).episode == "trajectory_0"   # numbering restarts per task
    assert found.binding(2).task == "cover_blocks"
    assert found.binding(5).task == "swap_blocks"


def test_unannotated_episodes_bind_to_nothing():
    # Most of the dataset has no annotation; that is normal, not an error.
    found = index()
    assert found.binding(0) is None and found.binding(1) is None
    assert len(found) == 5


def test_annotated_tasks_are_reported():
    assert index().annotated_tasks == ("cover_blocks", "swap_blocks")


def test_no_annotations_configured_binds_nothing():
    found = EpisodeIndex(tasks=(), episodes=EPISODES)
    assert len(found) == 0
    assert found.binding(2) is None
    assert found.directory(2) is None


def test_the_directory_is_only_known_with_a_root():
    assert index().directory(2) is None
    assert index(root="/data/annot").directory(2) == "/data/annot/cover_blocks/trajectory_0"
    assert index(root="/data/annot").directory(0) is None


# --- the failures this guards against --------------------------------------


def test_a_dataset_matching_no_task_at_all_is_an_error():
    # Published for another dataset, or its instructions were reworded: binding
    # nothing would train every episode with an empty memory, silently.
    with pytest.raises(ValueError, match="no episode of this dataset matches"):
        EpisodeIndex(tasks=TASKS, episodes=[(0, PLAIN, 500)])


def test_a_different_episode_count_is_an_error():
    with pytest.raises(ValueError, match="do not belong to this dataset"):
        EpisodeIndex(tasks=(Task("cover_blocks", COVER, 99),), episodes=EPISODES)


def test_a_task_absent_from_the_dataset_is_only_a_warning(caplog):
    # A subset of the dataset is a legitimate thing to train on.
    found = EpisodeIndex(
        tasks=TASKS, episodes=[(2, COVER, 540), (3, COVER, 541), (4, COVER, 542)])
    assert found.annotated_tasks == ("cover_blocks",)
    assert "swap_blocks" in caplog.text


def test_a_length_mismatch_is_an_error():
    found = index()
    with pytest.raises(ValueError, match="different length"):
        found.check_lengths({"cover_blocks/trajectory_0": 539})   # dataset says 540


def test_matching_lengths_pass():
    found = index()
    found.check_lengths({
        "cover_blocks/trajectory_0": 540, "cover_blocks/trajectory_1": 541,
        "cover_blocks/trajectory_2": 542,
        "swap_blocks/trajectory_0": 430, "swap_blocks/trajectory_1": 431,
    })


def test_an_annotation_without_a_recorded_length_is_not_checked():
    index().check_lengths({})   # nothing to compare against


# --- the release file ------------------------------------------------------


def test_the_release_index_is_read(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({
        "schema": "robodojo_progress_annotations_v1",
        "tasks": [{"task": "cover_blocks", "instruction": COVER, "episodes": 3}],
    }), encoding="utf-8")
    found = EpisodeIndex.from_release(str(tmp_path), EPISODES)
    assert found.annotated_tasks == ("cover_blocks",)
    assert found.binding(3).episode == "trajectory_1"


def test_a_missing_release_index_says_what_to_point_at(tmp_path):
    with pytest.raises(FileNotFoundError, match="annotation release"):
        EpisodeIndex.from_release(str(tmp_path), EPISODES)


def test_no_annotation_root_is_allowed():
    assert len(EpisodeIndex.from_release(None, EPISODES)) == 0


def test_a_release_of_another_schema_is_refused(tmp_path):
    """What ``window`` means is stated by the release, so it has to be checked.

    The same field has carried another meaning in our own earlier files --- a
    detection window rather than an acceptance interval --- and reading one as
    the other would train the selector against intervals nobody annotated.
    """
    import json

    (tmp_path / "index.json").write_text(json.dumps(
        {"schema": "something_else", "tasks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="robodojo_progress_annotations_v1"):
        EpisodeIndex.from_release(str(tmp_path), [(0, "do it", 10)])


def test_a_release_of_the_known_schema_is_read(tmp_path):
    import json

    from wall_x.data.backends.lerobot.robodojo.episodes import SCHEMA

    (tmp_path / "index.json").write_text(json.dumps(
        {"schema": SCHEMA,
         "tasks": [{"task": "t", "instruction": "do it", "episodes": 1}]}),
        encoding="utf-8")
    index = EpisodeIndex.from_release(str(tmp_path), [(0, "do it", 10)])
    assert index.binding(0).task == "t"
