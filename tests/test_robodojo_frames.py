"""Which frames a sample reads, and the images it hands the model."""

import json

import numpy as np
import pytest
from PIL import Image

from wall_x.data.backends.lerobot.robodojo import annotations as annot
from wall_x.data.backends.lerobot.robodojo.frames import (
    Geometry,
    action_request,
    prepare_images,
    selection_request,
)
from wall_x.data.backends.lerobot.robodojo.prompts import IMAGE, action_prompt

RECIPE = Geometry()            # what the released checkpoint used
RAW = (640, 480)               # the dataset's frames


def annotation(tmp_path, keyframes, meta=None, version=14):
    directory = tmp_path / "swap_blocks" / "trajectory_0"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "progress.json").write_text(json.dumps(
        {"version": version, "keyframes": keyframes, "meta": meta or []}), encoding="utf-8")
    return annot.load(str(directory))


def frame(colour=0, size=RAW):
    return np.full((size[1], size[0], 3), colour, dtype=np.uint8)


# --- the size every image is resized to ------------------------------------


def test_the_recipe_resizes_the_dataset_frames_to_what_training_used():
    assert RECIPE.image_size(480, 640) == (336, 448)


def test_the_long_side_goes_to_the_configured_resolution():
    height, width = RECIPE.image_size(720, 1280)
    assert max(height, width) == RECIPE.resolution


def test_both_sides_land_on_the_processor_grid():
    for shape in ((480, 640), (720, 1280), (240, 240), (1080, 1920)):
        height, width = RECIPE.image_size(*shape)
        assert height % RECIPE.factor == 0 and width % RECIPE.factor == 0


def test_an_empty_image_is_refused():
    with pytest.raises(ValueError, match="cannot be"):
        RECIPE.image_size(0, 640)


# --- which frames an action sample reads -----------------------------------


def test_an_action_sample_reads_the_keyframes_recorded_so_far(tmp_path):
    request = action_request(annotation(tmp_path, [10, 50, 90]), 60, RECIPE)
    assert request.recalled == (10, 50)
    assert request.current == 60
    assert request.candidates == ()


def test_recent_frames_follow_the_recalled_ones(tmp_path):
    geometry = Geometry(short_term=2, short_term_stride=4)
    request = action_request(annotation(tmp_path, [10]), 60, geometry)
    # Described as memory entries too, which is how the checkpoint was trained.
    assert request.recalled == (10, 52, 56)


def test_an_episode_without_annotation_reads_only_the_anchor(tmp_path):
    request = action_request(annot.EMPTY, 60, RECIPE)
    assert request.recalled == () and request.all_frames == (60,)


def test_capacity_caps_what_a_prompt_carries(tmp_path):
    request = action_request(annotation(tmp_path, list(range(0, 100, 2))), 99,
                             Geometry(capacity=3))
    assert request.recalled == (94, 96, 98)


# --- which frames a selection sample reads ---------------------------------


def test_a_selection_sample_reads_memory_then_candidates(tmp_path):
    request = selection_request(annotation(tmp_path, [10, 60]), 100,
                                Geometry(window=48, candidates=8))
    assert request.recalled == (10,)
    assert len(request.candidates) == 8
    assert request.head_frames == request.recalled + request.candidates


def test_selection_frames_are_clamped_into_the_episode(tmp_path):
    request = selection_request(annotation(tmp_path, []), 100,
                                Geometry(window=48, candidates=8), last_frame=70)
    assert max(request.candidates) <= 70


# --- the images -------------------------------------------------------------


def test_images_come_in_the_order_the_prompt_describes_them(tmp_path):
    request = action_request(annotation(tmp_path, [10, 50]), 60, RECIPE)
    head = {10: frame(10), 50: frame(50)}
    cameras = [frame(60), frame(61), frame(62)]
    images = prepare_images(request, head, cameras, RECIPE)
    assert [int(np.asarray(i)[0, 0, 0]) for i in images] == [10, 50, 60, 61, 62]


def test_every_image_is_resized_to_the_training_size(tmp_path):
    request = action_request(annotation(tmp_path, [10]), 60, RECIPE)
    images = prepare_images(request, {10: frame()}, [frame()] * 3, RECIPE)
    assert {(i.height, i.width) for i in images} == {(336, 448)}


def test_the_image_count_matches_the_prompt(tmp_path):
    # The contract that keeps image features on the right tokens.
    request = action_request(annotation(tmp_path, [10, 20, 30]), 60, RECIPE)
    head = {f: frame(f) for f in request.recalled}
    images = prepare_images(request, head, [frame()] * 3, RECIPE)
    prefix, _ = action_prompt("go", ("a", "b", "c"), action_horizon=4,
                              num_memory=len(request.recalled))
    assert prefix.count(IMAGE) == len(images)


def test_a_frame_described_but_not_read_is_an_error(tmp_path):
    request = action_request(annotation(tmp_path, [10, 50]), 60, RECIPE)
    with pytest.raises(KeyError, match="was not read"):
        prepare_images(request, {10: frame()}, [frame()] * 3, RECIPE)


def test_a_recalled_frame_is_cropped_to_its_object(tmp_path):
    # The crop keeps the object and drops the rest of the demonstration scene.
    a = annotation(tmp_path, [10], meta=[{"frame": 10, "box": [100, 100, 200, 200]}])
    geometry = Geometry(box_crop=True)
    request = action_request(a, 60, geometry)
    assert request.boxes == {10: (100, 100, 200, 200)}
    scene = np.zeros((480, 640, 3), dtype=np.uint8)
    scene[100:200, 100:200] = 255        # the object
    images = prepare_images(request, {10: scene}, [frame()] * 3, geometry, raw_size=RAW)
    recalled = np.asarray(images[0])
    assert recalled.shape[:2] == (336, 448)
    # After cropping, the object fills much more of the frame than it did.
    assert recalled.mean() > scene.mean() * 3


def test_candidates_and_the_current_observation_are_never_cropped(tmp_path):
    a = annotation(tmp_path, [10], meta=[{"frame": 10, "box": [100, 100, 200, 200]}])
    geometry = Geometry(box_crop=True, window=48, candidates=4)
    request = selection_request(a, 100, geometry)
    assert set(request.boxes) <= set(request.recalled)
    assert not set(request.boxes) & set(request.candidates)


def test_box_cropping_is_off_unless_asked(tmp_path):
    a = annotation(tmp_path, [10], meta=[{"frame": 10, "box": [100, 100, 200, 200]}])
    assert action_request(a, 60, RECIPE).boxes == {}


def test_images_accept_pil_as_well_as_arrays(tmp_path):
    request = action_request(annotation(tmp_path, [10]), 60, RECIPE)
    head = {10: Image.fromarray(frame(10))}
    images = prepare_images(request, head, [Image.fromarray(frame())] * 3, RECIPE)
    assert len(images) == 4 and all(isinstance(i, Image.Image) for i in images)
