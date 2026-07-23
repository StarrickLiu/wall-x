"""Qwen2.5 Torch executor used by environment evaluation.

The executor owns checkpoint/model/processor/normalizer initialization and
batched model execution. Environment adapters only select the training-model
adapter and translate executor results to environment actions.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from qwen_vl_utils.vision_process import smart_resize

from wall_x._vendor.x2robot_utils.text_templates import (
    get_prologue_with_embodied_information,
    preprocesser_call,
)

from wall_x._vendor.harrix.utils.ckpt_load import load_state_dict
from wall_x._vendor.harrix.utils.normalizer import build_normalizers
from wall_x._vendor.harrix.utils.train_config import (
    build_data_config,
    build_model_config,
    load_train_config_with_ckpt_overlay,
)
from wall_x.trainer.trainer_utils import load_wallx_processors_from_cfg


_SUPPORTED_MODES = {"flow", "ar", "dllm", "vqa", "subtask"}
_IMPLEMENTED_MODES = {"flow"}

# Special tokens used by the Qwen-VL action model.
_ROLE_START = "<|im_start|>"
_ROLE_END = "<|im_end|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"
_PROPRI = "<|propri|>"
_ACTION = "<|action|>"
logger = logging.getLogger(__name__)


_PUBLIC_CAMERA_LABELS = {
    "face_view": "front view",
    "right_wrist_view": "right wrist view",
    "left_wrist_view": "left wrist view",
}


def _camera_label(cam_name: str) -> str:
    return _PUBLIC_CAMERA_LABELS.get(cam_name, cam_name.replace("_", " "))


def _normalizer_width(normalizer, norm_key: str) -> int | None:
    if normalizer is None or norm_key not in normalizer.delta:
        return None
    return int(normalizer.delta[norm_key].shape[0])


def _normalize_real_prefix(normalizer, tensor, dataset_names):
    if normalizer is None:
        return tensor
    widths = [_normalizer_width(normalizer, name) for name in dataset_names]
    if any(width is None for width in widths) or len(set(widths)) != 1:
        return normalizer.normalize_data(tensor, dataset_names)
    width = widths[0]
    if tensor.shape[-1] == width:
        return normalizer.normalize_data(tensor, dataset_names)
    if tensor.shape[-1] < width:
        raise ValueError(
            f"normalizer width {width} exceeds tensor dim {tensor.shape[-1]}"
        )
    out = tensor.clone()
    out[..., :width] = normalizer.normalize_data(
        tensor[..., :width], dataset_names
    )
    out[..., width:] = 0
    return out


class Qwen2_5EvalExecutor:
    """Checkpoint-backed Qwen2.5 batched evaluator."""

    def __init__(self, cfg: Any, training_adapter) -> None:
        mode = cfg.model.action_mode
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"{type(self).__name__}: unknown action_mode={mode!r}, "
                f"supported={sorted(_SUPPORTED_MODES)}"
            )
        if mode not in _IMPLEMENTED_MODES:
            raise NotImplementedError(
                f"{type(self).__name__}: action_mode={mode!r} is not implemented; "
                "currently supported="
                f"{sorted(_IMPLEMENTED_MODES)}"
            )
        self._action_mode = mode

        ta = training_adapter
        device = "cuda"
        self._device = device

        # 1) typed cfg through the single wall_x loader (registers the data
        # backend, applies ckpt overlays).
        typed_cfg, train_config = load_train_config_with_ckpt_overlay(
            cfg.model.train_config_path, cfg.model.checkpoint_path
        )
        self._typed_cfg = typed_cfg

        # 2) data_config. Image resizing needs resolution/image_factor/min/max.
        self._data_config = build_data_config(typed_cfg, train_config)

        # 3) HF model config
        ConfigClass = ta.config_class()
        self._model_config = build_model_config(
            ConfigClass, cfg.model.checkpoint_path, typed_cfg, train_config
        )

        # 4) normalizers, including norm_key fallback.
        # Built before load_wallx_processors: OSS wall-x (refactor_2604) requires
        # an explicit normalizer for the action tokenizer; older builds accept it
        # via the same kwarg.
        normalizer_action, normalizer_propri, resolved_norm_key = build_normalizers(
            cfg.model.checkpoint_path, typed_cfg, cfg.model.norm_key
        )
        self._normalizer_action = normalizer_action
        self._normalizer_propri = normalizer_propri
        self._norm_key = resolved_norm_key

        # 5) processor + tokenizer_mixin (from the typed cfg, so ckpt
        # processor/tokenizer overlays are honored).
        procs = load_wallx_processors_from_cfg(
            typed_cfg, normalizer=normalizer_action, device=device
        )
        self._processor = procs["processor"]
        self._tokenizer_mixin = procs.get("tokenizer_mixin")

        # 6) model — the OSS package evaluates on the vanilla wall-x
        # model class (the internal opt inference engine is not part of
        # this release). The fused CUDA kernels are compiled at install
        # time and used automatically, so flow-inference accuracy
        # matches the internal build.
        ModelClass = ta.inference_model_class()
        self._model_class = ModelClass
        model = ModelClass(self._model_config, self._processor, self._tokenizer_mixin)
        model.resize_token_embeddings(len(self._processor.tokenizer))
        model.to_bfloat16_for_selected_params()

        # 7) checkpoint weights and model finalization.
        state_dict = load_state_dict(cfg.model.checkpoint_path, ModelClass)
        embed_key = "model.embed_tokens.weight"
        if embed_key in state_dict:
            ckpt_vocab = state_dict[embed_key].shape[0]
            cur_vocab = model.model.embed_tokens.weight.shape[0]
            if cur_vocab != ckpt_vocab:
                logger.info(
                    "resize_token_embeddings from %d to %d to match checkpoint",
                    cur_vocab,
                    ckpt_vocab,
                )
                model.resize_token_embeddings(ckpt_vocab)
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "%s load_state_dict: missing=%s unexpected=%s",
            type(self).__name__,
            len(msg.missing_keys),
            len(msg.unexpected_keys),
        )
        model.set_normalizer(
            copy.deepcopy(normalizer_action),
            copy.deepcopy(normalizer_propri),
        )
        model.eval()
        model.to(device)
        model.to_bfloat16_for_selected_params()
        self._model = model

        # 8) cached runtime fields
        self._cam_names = list(cfg.model.cam_names)
        self._action_horizon = int(cfg.model.action_horizon)
        self._action_dim = sum(typed_cfg.task.dof_config.values())
        self._num_inference_timesteps = 10
        self._robot_id = "10000"

    # ---- BaseInferAdapter API ----

    @property
    def chunk_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_mode(self) -> str:
        return self._action_mode

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def task_config(self):
        return self._typed_cfg.task

    def infer_batch(self, payloads: list[dict]) -> list[np.ndarray]:
        if self._action_mode == "flow":
            return self._flow_batch(payloads)
        raise NotImplementedError(
            f"{type(self).__name__}.infer_batch: action_mode={self._action_mode!r} "
            "is not dispatched"
        )

    @property
    def metadata(self):
        return {
            "model_type": "qwen2_5",
            "action_mode": self._action_mode,
            "action_horizon": self._action_horizon,
        }

    # ---- batched flow inference ----

    def _flow_batch(self, payloads: list[dict]) -> list[np.ndarray]:
        # 1) Collect model-ready observations.
        observations: list[dict] = []
        instructions: list[str] = []
        noises: list[np.ndarray | None] = []
        any_noise = False
        for p in payloads:
            observations.append(p["observation"])
            instructions.append(p["instruction"])
            n = p.get("noise")
            if n is not None:
                any_noise = True
            noises.append(n)

        # 2) Stack noise when provided; all-None delegates sampling to the model.
        if any_noise:
            if any(n is None for n in noises):
                raise ValueError(
                    "payload noise must be provided for every payload or for none"
                )
            batch_noise = torch.stack(
                [torch.from_numpy(n).to(dtype=torch.float32) for n in noises], dim=0
            )
        else:
            batch_noise = None

        # 3) Prompts and model inputs.
        prefix_list, postfix_list = [], []
        for ins in instructions:
            prefix, postfix = self._get_flow_prompt(ins)
            prefix_list.append(prefix)
            postfix_list.append(postfix)
        batch_inputs = self._construct_model_input(
            observations, prefix_list, postfix_list
        )

        # 4) Normalized zero action as the flow starting point.
        padding = torch.zeros(
            (
                len(batch_inputs["dataset_names"]),
                1,
                self._action_dim,
            ),
            dtype=torch.float32,
        )
        padding_action = _normalize_real_prefix(
            self._normalizer_action, padding, batch_inputs["dataset_names"]
        ).to(batch_inputs["input_ids"].device)

        # 5) Flow forward.
        model_output = self._model.generate_flow_action(
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            num_inference_timesteps=self._num_inference_timesteps,
            padding_action=padding_action,
            noise=batch_noise,
            **batch_inputs,
        )

        # 6) Return one raw model action chunk per request.
        predict_action = model_output["predict_action"]  # (B, H, D_action)
        if isinstance(predict_action, torch.Tensor):
            predict_action = predict_action.detach().cpu().numpy()
        return [predict_action[i : i + 1] for i in range(len(payloads))]

    # ---- prompt template ----

    def _get_flow_prompt(self, instruction: str) -> tuple[str, str]:
        """Build the flow-action prompt."""
        if getattr(self._typed_cfg.data, "use_embodied_system_prompt_ratio", 0) > 0:
            robot_id = self._robot_id if self._norm_key in ("x2_normal", "ex_normal") else 0
            cam_name_mapping = {cn: cn for cn in self._cam_names}
            prologue = get_prologue_with_embodied_information(
                dataset_name=self._norm_key,
                cam_mapping=cam_name_mapping,
                robot_id=robot_id,
                uid="",
                config=self._data_config,
            )
        else:
            prologue = (
                f"{_ROLE_START}system\nYou are a helpful assistant.{_ROLE_END}\n"
            )
        user_request = f"{_ROLE_START}user\nObservation:"
        for cn in self._cam_names:
            user_request += (
                f" {_camera_label(cn)}: "
                f"{_VISION_START}{_IMAGE_PAD}{_VISION_END}"
            )
        user_request += "\nInstruction:"
        text_prompt = (
            f"\nPredict the next action in robot action.\nProprioception: {_PROPRI}\n"
        )
        user_message = (
            f"{user_request} {instruction}{text_prompt}{_ROLE_END}\n"
        )
        assistant_message = f"{_ROLE_START}assistant\n"
        flow_action = _ACTION * self._action_horizon

        prefix_text = prologue + user_message + assistant_message
        postfix_text = flow_action
        return prefix_text, postfix_text

    # ---- batched model input construction ----

    def _construct_model_input(
        self,
        observations: list[dict],
        prefix_list: list[str],
        postfix_list: list[str],
    ) -> dict:
        """Build model inputs for the flow inference path."""
        batch_size = len(observations)
        dataset_names = [self._norm_key] * batch_size

        # Proprioception and masks are supplied as model-ready ndarrays.
        agent_pos = torch.cat(
            [torch.from_numpy(o["proprioception"]) for o in observations], dim=0
        )
        agent_pos_mask = torch.cat(
            [torch.from_numpy(o["agent_pos_mask"]) for o in observations], dim=0
        )
        dof_mask = torch.cat(
            [torch.from_numpy(o["dof_mask"]) for o in observations], dim=0
        )
        agent_pos = _normalize_real_prefix(
            self._normalizer_propri, agent_pos, dataset_names
        )

        # Resize images per sample.
        image_inputs: list[torch.Tensor] = []
        all_image_sizes: list[tuple[int, int]] = []
        for o in observations:
            for cn in self._cam_names:
                if cn not in o:
                    continue
                tensor = self._resize_image(o[cn], cn)
                image_inputs.append(tensor)
                # Tensor (H, W, C) to PIL-compatible (W, H).
                all_image_sizes.append((tensor.shape[1], tensor.shape[0]))

        inputs = preprocesser_call(
            processor=self._processor,
            prefix_text=prefix_list,
            postfix_text=postfix_list,
            images=image_inputs,
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1000,
            pad_to_128_multiple=False,
            pad_prefix_to_same_length=False,
            norm_state=(
                agent_pos
                if self._typed_cfg.task.use_state_string_representation
                else None
            ),
            agent_pos_mask=agent_pos_mask,
            state_augmentation_prob=0.0,
            state_drop_prob=0.0,
            state_augmentation_ratio=0.0,
            state_bins=getattr(self._typed_cfg.data, "state_bins", 256),
            inference_mode=True,
        )

        action_token_id = self._processor.tokenizer.convert_tokens_to_ids(_ACTION)
        moe_token_types = inputs["input_ids"] == action_token_id

        extra = {
            "proprioception": agent_pos.detach(),
            "agent_pos_mask": agent_pos_mask,
            "dof_mask": dof_mask,
            "image_size": all_image_sizes,
            "moe_token_types": moe_token_types,
            "dataset_names": dataset_names,
        }
        inputs.update(extra)
        return _move_to_device(inputs, self._device)

    # ---- image resizing ----

    def _resize_image(self, img: np.ndarray, cam_name: str) -> torch.Tensor:
        """Resize one image with the train-time image config."""
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil = img
        else:
            raise ValueError(f"unsupported image type: {type(img)}")
        orig_w, orig_h = pil.size

        target = self._data_config.resolution.get(cam_name, -1)
        if target != -1:
            if orig_w > orig_h:
                new_w, new_h = target, int(target * orig_h / orig_w)
            else:
                new_h, new_w = target, int(target * orig_w / orig_h)
            pil = pil.resize((new_w, new_h))

        cur_w, cur_h = pil.size
        resized_h, resized_w = smart_resize(
            cur_h,
            cur_w,
            factor=self._data_config.image_factor,
            min_pixels=self._data_config.min_pixels,
            max_pixels=self._data_config.max_pixels,
        )
        resized = pil.resize((resized_w, resized_h))
        return torch.from_numpy(np.array(resized)).to(self._device)


def _move_to_device(obj, device):
    """Recursively move tensors inside common containers to ``device``."""
    from transformers import BatchFeature

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (dict, BatchFeature)):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj
