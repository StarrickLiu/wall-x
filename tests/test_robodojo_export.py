"""Reading frames out of a LeRobot dataset.

Built against the real library on a dataset written for the test, so the thing
under test is the contract with LeRobot rather than a description of it.
"""

import numpy as np
import pytest

from wall_x.data.backends.lerobot.robodojo.export import (
    ACTION_KEY,
    CAMERA_KEYS,
    STATE_KEY,
    LeRobotFrameSource,
    _to_hwc_uint8,
    episodes_of,
)
from wall_x.data.backends.lerobot.robodojo.dataset import Episode

lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")

HEIGHT, WIDTH = 32, 48
LENGTHS = (7, 5)
TASKS = ("remember the colours", "sort the objects")


def pose(episode, frame):
    """A pose whose numbers say which episode and frame it came from."""
    arm = [episode, frame, 0.0, 1.0, 0.0, 0.0, 0.0, frame / 100]
    return np.array(arm + arm, dtype=np.float32)


def picture(episode, frame, camera):
    """A frame whose first pixel says which frame and camera it is."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[0, 0] = (episode, frame, camera)
    return image


class Columns:
    """The column view of an episode table, as LeRobot's metadata exposes it."""

    def __init__(self, rows):
        self.column_names = list(rows[0])
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, name):
        return [row[name] for row in self._rows]


class Opened:
    """A written dataset, read through the real library.

    Frames and poses come from the library itself. Only the episode table
    stands in: opening a dataset by name makes the library ask the hub about a
    repository, which a test must not do. The table logic is checked against
    the official export's own metadata instead (see the parity scripts).
    """

    def __init__(self, data, lengths):
        self._data = data
        start = 0
        rows = []
        for index, length in enumerate(lengths):
            rows.append({"episode_index": index, "tasks": [TASKS[index]],
                         "length": length, "dataset_from_index": start,
                         "dataset_to_index": start + length})
            start += length
        self.meta = type("Meta", (), {"episodes": Columns(rows),
                                      "features": data.meta.features})()

    def __getitem__(self, row):
        return self._data[row]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("robodojo_ee")
    features = {
        STATE_KEY: {"dtype": "float32", "shape": (16,), "names": None},
        ACTION_KEY: {"dtype": "float32", "shape": (16,), "names": None},
    }
    for key in CAMERA_KEYS:
        features[key] = {"dtype": "image", "shape": (HEIGHT, WIDTH, 3),
                         "names": ["height", "width", "channels"]}
    data = lerobot.LeRobotDataset.create(
        repo_id="test/robodojo_ee", fps=25, features=features, root=root / "ds",
        use_videos=False)
    for episode, length in enumerate(LENGTHS):
        for frame in range(length):
            row = {STATE_KEY: pose(episode, frame),
                   ACTION_KEY: pose(episode, frame) * 2,
                   "task": TASKS[episode]}
            for camera, key in enumerate(CAMERA_KEYS):
                row[key] = picture(episode, frame, camera)
            data.add_frame(row)
        data.save_episode()
    return Opened(data, LENGTHS)


@pytest.fixture
def source(dataset):
    return LeRobotFrameSource(dataset)


# --- the episode table -----------------------------------------------------


def test_episodes_are_reported_with_their_instruction_and_length(source):
    episodes = source.episodes()
    assert [e.index for e in episodes] == [0, 1]
    assert [e.length for e in episodes] == list(LENGTHS)
    assert [e.instruction for e in episodes] == list(TASKS)


def test_a_frame_addresses_the_row_its_episode_starts_at(source):
    assert source.row(0, 0) == 0
    assert source.row(0, LENGTHS[0] - 1) == LENGTHS[0] - 1
    assert source.row(1, 0) == LENGTHS[0]          # the next episode starts after it


def test_a_frame_outside_its_episode_is_refused(source):
    with pytest.raises(IndexError, match="outside it"):
        source.row(0, LENGTHS[0])
    with pytest.raises(IndexError):
        source.row(0, -1)


def test_an_unknown_episode_is_refused(source):
    with pytest.raises(KeyError, match="no episode 9"):
        source.row(9, 0)


# --- frames ----------------------------------------------------------------


def test_the_head_image_is_the_frame_that_was_written(source):
    for episode, length in enumerate(LENGTHS):
        for frame in (0, length - 1):
            got = source.head_image(episode, frame)
            assert got.shape == (HEIGHT, WIDTH, 3)
            assert tuple(got[0, 0]) == (episode, frame, 0)


def test_every_camera_comes_back_in_order(source):
    images = source.camera_images(1, 2)
    assert [tuple(i[0, 0]) for i in images] == [(1, 2, 0), (1, 2, 1), (1, 2, 2)]


def test_the_head_image_is_the_first_camera(source):
    assert np.array_equal(source.head_image(1, 2), source.camera_images(1, 2)[0])


# --- poses -----------------------------------------------------------------


def test_the_state_is_the_pose_that_was_written(source):
    assert np.allclose(source.state(1, 3), pose(1, 3), atol=1e-6)


def test_commands_start_at_the_frame_asked_for(source):
    commands = source.commands(0, 2, 3)
    assert len(commands) == 3
    assert np.allclose(commands[0], pose(0, 2) * 2, atol=1e-6)
    assert np.allclose(commands[2], pose(0, 4) * 2, atol=1e-6)


def test_commands_stop_at_the_end_of_the_episode(source):
    # They must not run into the next episode's rows.
    commands = source.commands(0, LENGTHS[0] - 2, 8)
    assert len(commands) == 2
    assert np.allclose(commands[-1], pose(0, LENGTHS[0] - 1) * 2, atol=1e-6)


# --- what the dataset must provide -----------------------------------------


def test_a_dataset_without_the_expected_cameras_is_refused(dataset):
    with pytest.raises(KeyError, match="has no"):
        LeRobotFrameSource(dataset, camera_keys=("observation.images.nope",))


# --- image conversion ------------------------------------------------------


def test_a_float_channels_first_frame_becomes_hwc_bytes():
    frame = np.zeros((3, 4, 5), dtype=np.float32)
    frame[0, 0, 0] = 1.0
    out = _to_hwc_uint8(frame)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    assert out[0, 0, 0] == 255


def test_a_byte_frame_is_left_as_it_is():
    frame = np.full((4, 5, 3), 7, dtype=np.uint8)
    assert np.array_equal(_to_hwc_uint8(frame), frame)


# --- the whole chain -------------------------------------------------------


def test_samples_come_out_of_a_real_lerobot_dataset(dataset, tmp_path):
    """From a LeRobot dataset to a training sample, through every layer."""
    import json

    from wall_x.data.backends.lerobot.robodojo.dataset import ACTION, RoboDojoDataset
    from wall_x.data.backends.lerobot.robodojo.episodes import SCHEMA, EpisodeIndex
    from wall_x.data.backends.lerobot.robodojo.frames import Geometry

    # Annotate the first episode: frame 2 recorded progress.
    directory = tmp_path / "remember" / "trajectory_0"
    directory.mkdir(parents=True)
    (directory / "progress.json").write_text(json.dumps(
        {"version": 14, "keyframes": [2], "total_frames": LENGTHS[0]}), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps(
        {"schema": SCHEMA,
         "tasks": [{"task": "remember", "instruction": TASKS[0],
                    "episodes": 1}]}),
        encoding="utf-8")

    source = LeRobotFrameSource(dataset)
    index = EpisodeIndex.from_release(
        str(tmp_path), [(e.index, e.instruction, e.length) for e in source.episodes()])
    index.check_lengths({"remember/trajectory_0": LENGTHS[0]})
    data = RoboDojoDataset(source, index, geometry=Geometry(window=4, candidates=4),
                           action_horizon=2)

    action = next(data[i] for i in range(len(data))
                  if data[i]["kind"] == ACTION and data[i]["frame"] == 4
                  and data[i]["episode"] == 0)
    # Frame 2 is a keyframe recorded before frame 4, so it is recalled.
    assert action["frames_read"] == (2, 4)
    assert action["text"].count("[M_0]") == 1
    assert len(action["images"]) == 4                  # one recalled, three cameras
    assert tuple(np.asarray(action["images"][0])[0, 0])[:2] == (0, 2)
    assert action["text"].count("<|vision_start|>") == len(action["images"])
    assert action["action"].shape == (2, 20)
    # The poses move by one unit of episode index per frame on the left arm.
    assert np.allclose(action["agent_pos"][0], 0.0, atol=1e-6)

    selection = next(data[i] for i in range(len(data))
                     if data[i]["kind"] != ACTION and data[i]["frame"] == 4)
    assert "<|kf_write|>" in selection["text"]
    assert selection["text"].count("<|vision_start|>") == len(selection["images"])


class FakeTable:
    """An episode table that answers the way the metadata parquet does."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, column):
        return [row[column] for row in self._rows]


class Arrowish:
    """A value that only yields its contents through ``as_py``."""

    def __init__(self, value):
        self._value = value

    def as_py(self):
        return self._value


class FakeMeta:
    def __init__(self, table):
        self.episodes = table


def test_episodes_of_reads_the_table_without_opening_the_videos():
    meta = FakeMeta(FakeTable([
        {"episode_index": 0, "tasks": ["pick it up"], "length": 240},
        {"episode_index": 1, "tasks": ["pick it up"], "length": 241},
    ]))
    assert episodes_of(meta) == (
        Episode(0, "pick it up", 240), Episode(1, "pick it up", 241))


@pytest.mark.parametrize("stored", [
    "pick it up",
    ["pick it up"],
    ("pick it up", ),
    Arrowish(["pick it up"]),
    Arrowish("pick it up"),
])
def test_the_task_string_survives_however_the_table_stores_it(stored):
    # A wrong read here is silent: the instruction is what binds an episode to
    # its annotation, so "['pick it up']" would simply match nothing.
    meta = FakeMeta(FakeTable([{"episode_index": 0, "tasks": stored, "length": 7}]))
    assert episodes_of(meta)[0].instruction == "pick it up"


def test_an_episode_with_no_task_string_is_an_error():
    meta = FakeMeta(FakeTable([{"episode_index": 0, "tasks": [], "length": 7}]))
    with pytest.raises(ValueError, match="no task string"):
        episodes_of(meta)
