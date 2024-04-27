import os
import torch
torch.set_printoptions(sci_mode=False)

import tempfile
import tensorrt as trt

from utils import calculate_time
from llava_vision_pth import LlavaVision
from tensorrt_llm.runtime import Session, TensorInfo
from tensorrt_llm._utils import trt_dtype_to_torch


def export_onnx(model: torch.nn.Module, image: torch.Tensor,
                output_dir: str) -> None:
    torch.onnx.export(model,
                      image,
                      f"{output_dir}/tmp.onnx",
                      opset_version=17,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {
                          0: "batch"
                      }})


def export_engine(img_h: int, img_w: int, max_batch_size: int, onnx_path: str,
                  engine_path: str) -> None:
    logger = trt.Logger(trt.Logger.ERROR)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as model:
        parser.parse(model.read(), onnx_path)

    # delete onnx model
    os.remove(onnx_path)

    nBS = -1
    nMinBS = 1
    nOptBS = max(nMinBS, int(max_batch_size / 2))
    nMaxBS = max_batch_size

    inputT = network.get_input(0)
    inputT.shape = [nBS, 3, img_h, img_w]
    profile.set_shape(inputT.name, [nMinBS, 3, img_h, img_w],
                      [nOptBS, 3, img_h, img_w], [nMaxBS, 3, img_h, img_w])
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    assert engine is not None, "build engine failed"
    with open(engine_path, 'wb') as f:
        f.write(engine)


class TrtSession:

    def __init__(self, engine_path: str) -> None:
        super().__init__()

        self.stream = torch.cuda.current_stream().cuda_stream

        # load engine
        with open(engine_path, "rb") as f:
            engine = f.read()
        self.session = Session.from_serialized_engine(engine)

    @calculate_time
    def infer(self, image: torch.Tensor) -> torch.Tensor:
        vis_inputs = {"input": image}
        vis_outputs = self.session.infer_shapes(
            [TensorInfo("input", trt.DataType.HALF, image.shape)])
        vis_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device="cuda")
            for t in vis_outputs
        }

        ok = self.session.run(vis_inputs, vis_outputs, self.stream)
        assert ok
        torch.cuda.synchronize()

        return vis_outputs["output"]


if __name__ == "__main__":
    model_dir = "/data/models/llava-v1.5-7b"
    model = LlavaVision(model_dir=model_dir)

    image = model.load_image("./images", num=4)

    max_batch_size = 8
    force = False
    engine_path = os.path.join(model_dir + "-vision",
                               f"vision_{max_batch_size}.engine")
    if not os.path.exists(engine_path) or force:
        tmp_dir = tempfile.TemporaryDirectory()
        export_onnx(model, image, tmp_dir.name)
        export_engine(image.shape[2],
                      image.shape[3],
                      max_batch_size=max_batch_size,
                      onnx_path=f"{tmp_dir.name}/tmp.onnx",
                      engine_path=engine_path)

    trt_session = TrtSession(engine_path=engine_path)

    for _ in range(10):
        x = trt_session.infer(image)
    print("out:\n", x[0].cpu().numpy())
    print("out shape: ", x.shape)
