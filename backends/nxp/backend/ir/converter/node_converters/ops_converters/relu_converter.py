# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    torch_type_to_numpy_type,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    _is_dequant_node,
    CustomDelegationOptions,
    is_not_qdq_node,
    NodeConverter,
    Partition,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import quantize
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
)
from executorch.backends.nxp.backend.neutron_operator_support import (
    activation_supported_on_target,
    NeutronTargetSpec,
)
from torch.fx import Node
from torch.nn import Parameter


class ReLUConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        return activation_supported_on_target(node)

    @staticmethod
    def is_relu_preserved_under_quantization(
        node: Node, min_val: int = 0, max_val: int | None = None
    ) -> bool:
        q_node = node.args[0]

        if not _is_dequant_node(q_node):
            return False

        if len(q_node.args) == 6:
            # per-tensor
            _, scale, zp, quant_min, quant_max, q_type = q_node.args
        else:
            # per-channel
            _, scale, zp, quant_min, quant_max, _, q_type = q_node.args

        q_type = torch_type_to_numpy_type(q_type).type
        quantized_min_val = quantize(
            value=min_val,
            zero_point=zp,
            scale=scale,
            quant_min=quant_min,
            quant_max=quant_max,
            dtype=q_type,
        )

        if max_val:
            quantized_max_val = quantize(
                value=max_val,
                zero_point=zp,
                scale=scale,
                quant_min=quant_min,
                quant_max=quant_max,
                dtype=q_type,
            )
            return (
                # At least one bound is in quantization range
                (quant_min < quantized_min_val)
                or (quantized_max_val < quant_max)
                # ReLU does not degrade to constant
                and not (quant_max < quantized_min_val)
                and not (quant_min > quantized_max_val)
            )

        return quant_min < quantized_min_val < quant_max

    @classmethod
    def supports_partitioning_result(
        cls,
        node: Node,
        partition_list: list[Partition],
        custom_delegation_options: CustomDelegationOptions,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        is_alone_in_partition = cls.is_node_alone_in_partition(
            node, partition_list, filter_fn=is_not_qdq_node
        )
        if is_alone_in_partition:
            return cls.is_relu_preserved_under_quantization(node)

        return True

    def convert(self, node: Node):
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        t_op.opcode_index = self.builder.op_code_index_for_op_type(BuiltinOperator.RELU)

        self.builder.append_operators([t_op])
