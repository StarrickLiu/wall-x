"""RoboDojo training samples, end to end over a stand-in dataset."""

import json

import numpy as np
import pytest

from wall_x.data.backends.lerobot.robodojo.dataset import ACTION, SELECTION, Episode, RoboDojoDataset
from wall_x.data.backends.lerobot.robodojo.episodes import SCHEMA, EpisodeIndex, Task
from wall_x.data.backends.lerobot.robodojo.frames import Geometry
from wall_x.data.state_noise import StateNoise

COVER = "Cover the blocks, remember their colors, then uncover them."
PLAIN = "Sort the objects into the baskets."
RAW = (640, 480)


class Source:
    """A dataset of two episodes: one annotated, one not.

    Poses move along +x by a millimetre a frame, so actions are predictable,
    and every frame carries its own index as its pixel value.
    """

    def __init__(self, length=120):
        self.length = length
        self.reads = []

    def episodes(self):
        return [Episode(0, COVER, self.length), Episode(1, PLAIN, self.length)]

    def _pose(self, episode, frame):
        x = 0.001 * frame
        arm = [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]        # identity rotation
        return arm + [-x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    def head_image(self, episode, frame):
        self.reads.append((episode, frame))
        return np.full((RAW[1], RAW[0], 3), frame % 256, dtype=np.uint8)

    def camera_images(self, episode, frame):
        return [np.full((RAW[1], RAW[0], 3), c, dtype=np.uint8) for c in (1, 2, 3)]

    def state(self, episode, frame):
        return self._pose(episode, frame)

    def commands(self, episode, frame, count):
        # A real source has nothing past the end of the episode.
        return [self._pose(episode, frame + k)
                for k in range(count) if frame + k < self.length]


def annotations(tmp_path, keyframes=(10, 40, 70), meta=None, episodes=(0, 1)):
    directory = tmp_path / "cover_blocks" / "trajectory_0"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "progress.json").write_text(json.dumps(
        {"version": 14, "keyframes": list(keyframes), "meta": meta or [],
         "total_frames": 120}), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps(
        {"schema": SCHEMA,
         "tasks": [{"task": "cover_blocks", "instruction": COVER,
                    "episodes": 1}]}),
        encoding="utf-8")
    source = Source()
    return EpisodeIndex.from_release(str(tmp_path), [
        (e.index, e.instruction, e.length) for e in source.episodes()])


def find(data, kind, frame, episode=0):
    """The sample of one kind at one frame; samples are grouped per episode."""
    for i in range(len(data)):
        sample = data[i]
        if sample["kind"] == kind and sample["frame"] == frame \
                and sample["episode"] == episode:
            return sample
    raise AssertionError(f"no {kind} sample at frame {frame}")


def dataset(tmp_path, **kwargs):
    source = kwargs.pop("source", Source())
    return RoboDojoDataset(source, annotations(tmp_path, **kwargs.pop("annot", {})),
                           **kwargs), source


# --- what the dataset contains ---------------------------------------------


def test_every_frame_with_a_future_is_an_action_sample(tmp_path):
    data, _ = dataset(tmp_path, selection=False)
    # The last frame of an episode has no motion to ask for.
    assert len(data) == 238                   # two episodes of 120 frames, minus the last
    assert {data[i]["kind"] for i in (0, 150)} == {ACTION}


def test_only_annotated_episodes_get_selection_samples(tmp_path):
    data, _ = dataset(tmp_path)
    assert len(data) == 358                   # 238 action + 120 selection
    assert data.selection_samples == 120


def test_selection_samples_can_be_turned_off(tmp_path):
    data, _ = dataset(tmp_path, selection=False)
    assert data.selection_samples == 0


def test_a_stride_thins_the_anchors(tmp_path):
    data, _ = dataset(tmp_path, stride=10, selection=False)
    assert len(data) == 24   # 12 anchors per episode


def test_a_zero_stride_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        dataset(tmp_path, stride=0)


# --- action samples --------------------------------------------------------


def test_an_action_sample_recalls_the_keyframes_so_far(tmp_path):
    data, _ = dataset(tmp_path, selection=False)
    sample = data[50]                          # episode 0, frame 50
    assert sample["frames_read"] == (10, 40, 50)
    assert sample["text"].count("[M_") == 2
    assert len(sample["images"]) == 5           # two recalled plus three cameras


def test_an_unannotated_episode_trains_without_memory(tmp_path):
    data, _ = dataset(tmp_path, selection=False)
    sample = find(data, ACTION, 50, episode=1)
    assert sample["frames_read"] == (50,)
    assert "progress memory" not in sample["text"]
    assert len(sample["images"]) == 3


def test_the_action_chunk_has_one_entry_per_step(tmp_path):
    data, _ = dataset(tmp_path, selection=False, action_horizon=8)
    sample = data[50]
    assert sample["action"].shape == (8, 20)
    assert sample["action_valid_mask"].all()
    assert not np.isnan(sample["action"]).any()
    # A millimetre of +x per frame, and step k targets the frame after k.
    assert np.allclose(sample["action"][0][0], 0.001, atol=1e-9)
    assert np.allclose(sample["action"][7][0], 0.008, atol=1e-9)


def test_a_chunk_running_past_the_episode_end_is_marked_invalid(tmp_path):
    data, _ = dataset(tmp_path, selection=False, action_horizon=8)
    sample = find(data, ACTION, 115)
    assert sample["action_valid_mask"].tolist() == [True, True, True, True,
                                                    False, False, False, False]
    # Padding is NaN, which is how the collator learns to mask it out.
    assert np.isnan(sample["action"][4:]).all()
    assert not np.isnan(sample["action"][:4]).any()


def test_the_pose_is_left_for_the_collator_to_write(tmp_path):
    # The collator fills it from the pose it normalized, so the text and the
    # normalization cannot disagree.
    data, _ = dataset(tmp_path, selection=False)
    sample = data[50]
    assert sample["agent_pos"].shape == (20,)
    assert "Proprioception: <|propri|>" in sample["text"]


def test_the_image_count_always_matches_the_prompt(tmp_path):
    data, _ = dataset(tmp_path)
    for i in (0, 5, 50, 118, 125, 200, 250, 357):
        sample = data[i]
        blocks = sample["text"].count("<|vision_start|>")
        assert blocks == len(sample["images"]), (i, sample["kind"])


# --- selection samples -----------------------------------------------------


def test_a_selection_sample_asks_about_its_candidates(tmp_path):
    data, _ = dataset(tmp_path, geometry=Geometry(window=48, candidates=8))
    # At frame 100 the window covers 52..99, so two keyframes are already
    # memory and one is inside the window to be asked about.
    sample = find(data, SELECTION, 100)
    assert sample["kind"] == SELECTION
    assert sample["text"].count("<|R_") >= 8
    assert len(sample["images"]) == 10          # two recalled plus eight candidates


def test_a_selection_sample_with_a_keyframe_in_the_window_is_a_write(tmp_path):
    data, _ = dataset(tmp_path, geometry=Geometry(window=48, candidates=8))
    sample = find(data, SELECTION, 100)
    assert "<|kf_write|>" in sample["text"]
    assert sample["slot_ranges"]


def test_a_selection_sample_with_nothing_to_write_is_a_skip(tmp_path):
    # A four-frame window at 100 contains no keyframe.
    data, _ = dataset(tmp_path, geometry=Geometry(window=4, candidates=8))
    sample = find(data, SELECTION, 100)
    assert "<|kf_skip|>" in sample["text"]
    assert sample["slot_ranges"] == []


def test_an_answer_carries_the_object_box_when_the_task_annotates_one(tmp_path):
    data, _ = dataset(
        tmp_path, geometry=Geometry(window=48, candidates=8, box_crop=True),
        annot={"keyframes": (40,), "meta": [{"frame": 40, "box": [160, 120, 480, 360]}]})
    sample = find(data, SELECTION, 60)
    # Boxes travel as 0..1000 of the frame, so the quarter-to-three-quarter box
    # of a 640x480 frame reads as 250,250 to 750,750.
    assert "(250,250),(750,750)" in sample["text"]


# --- reading frames --------------------------------------------------------


def test_a_recalled_frame_is_read_once_however_often_it_is_recalled(tmp_path):
    data, source = dataset(tmp_path, selection=False)
    source.reads.clear()
    data[50]
    recalled = [read for read in source.reads if read[1] in (10, 40)]
    assert len(recalled) == 2


# --- the perturbation ------------------------------------------------------


def test_the_perturbation_moves_the_pose_the_model_is_shown(tmp_path):
    plain, _ = dataset(tmp_path, selection=False)
    noisy, _ = dataset(tmp_path, selection=False,
                       state_noise=StateNoise(position=0.01, rotation=0.05))
    assert not np.allclose(plain[50]["agent_pos"],
                           noisy[50]["agent_pos"])
    # The actions stay relative to the true pose.
    assert np.allclose(plain[50]["action"], noisy[50]["action"])


def test_without_the_perturbation_the_pose_is_the_measured_one(tmp_path):
    data, source = dataset(tmp_path, selection=False)
    sample = data[50]
    assert np.allclose(sample["agent_pos"][0], 0.05)   # 50 mm along +x


def test_the_perturbation_repeats_for_the_same_sample(tmp_path):
    data, _ = dataset(tmp_path, selection=False, state_noise=StateNoise(position=0.01))
    assert np.allclose(data[50]["agent_pos"],
                       data[50]["agent_pos"])
