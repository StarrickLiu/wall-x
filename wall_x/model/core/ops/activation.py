"""Activation fusion operators."""

import logging

import torch
import torch.nn.functional as F

from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)


class SiluAndMulOp(OpsProxy):
    """Fused SiLU + elementwise multiply.

    Signatures:
      - silu_and_mul(gate, up) -> silu(gate) * up
      - silu_and_mul(gate_up, inplace=True) -> silu(gate) * up

    In split mode, ``inplace=True`` is best-effort for inference. During
    autograd-enabled execution the PyTorch fallback returns a fresh tensor to
    avoid writing into a view.
    """

    @property
    def _external_accel_name(self):
        return "silu_and_mul"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import SiluAndMul

            return SiluAndMul()
        except ImportError:
            return None
        except Exception as e:
            logger.warning("SiluAndMulOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, input1, input2=None, inplace=False):
        if input2 is None:
            if input1.size(-1) % 2 != 0:
                raise AssertionError("last dim must be even for split mode")
            gate, up = input1.chunk(2, dim=-1)
            output = F.silu(gate) * up
            if inplace and not torch.is_grad_enabled():
                gate.copy_(output)
                return gate
            return output

        output = F.silu(input1) * input2
        if inplace:
            input1.copy_(output)
            return input1
        return output


silu_and_mul = SiluAndMulOp()
