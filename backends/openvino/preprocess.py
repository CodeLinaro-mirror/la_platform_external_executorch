# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from typing import final, List

from executorch.backends.openvino._passes import DecomposeFloorDividePass

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass
from openvino import compile_model  # type: ignore[import-untyped]
from openvino.frontend import FrontEndManager  # type: ignore[import-untyped]
from openvino.frontend.pytorch.fx_decoder import (  # type: ignore[import-untyped]
    TorchFXPythonDecoder,
)


@final
class OpenvinoBackend(BackendDetails):

    @classmethod
    def preprocess(
        cls, edge_program: ExportedProgram, module_compile_spec: List[CompileSpec]
    ) -> PreprocessResult:
        """
        Preprocesses the exported program and compiles it for the OpenVINO backend.

        Args:
            edge_program (ExportedProgram): The exported program representing the model.
            module_compile_spec (List[CompileSpec]): A list of compile specifications for the OpenVINO backend.

        Returns:
            PreprocessResult: The result of preprocessing, including the compiled model bytes.
        """
        for pass_cls in [DimOrderOpsRevertPass, DecomposeFloorDividePass]:
            result = pass_cls()(edge_program.graph_module)
            if result and result.graph_module:
                edge_program._graph_module = result.graph_module

        input_names = edge_program.graph_signature.user_inputs
        input_types = []
        for node in edge_program.graph.nodes:
            if node.target in input_names:
                input_types.append(node.meta["val"].dtype)

        compile_options = {}
        for spec in module_compile_spec:
            compile_options[spec.key] = spec.value.decode()

        # The converter chooses some lowerings (e.g. boolean-mask indexing) from
        # the input types, so they must be known before conversion starts.
        decoder = TorchFXPythonDecoder(edge_program.module(), input_types=input_types)
        frontend = FrontEndManager().load_by_framework("pytorch")
        compiled = compile_model(
            frontend.convert(frontend.load(decoder)),
            compile_options.get("device", "CPU"),
        )
        model_bytes = compiled.export_model()

        return PreprocessResult(processed_bytes=model_bytes.getvalue())
