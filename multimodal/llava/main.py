import os
import torch

from llava_vision_pth import LlavaVision
from llava_vision_trt import export_onnx, export_engine, TrtSession
from llava_vision_onnx import OnnxSession

if __name__ == "__main__":
    model_dir = "/data/models/llava-v1.5-7b"
    dtype = torch.float32

    # initialize torch model and load image
    model = LlavaVision(model_dir=model_dir, dtype=dtype)
    image = model.load_image("./images", num=1)

    # export engien when necessary
    max_batch_size = 8
    force = True
    engine_path = os.path.join(model_dir + "-vision",
                               f"vision_{max_batch_size}.engine")
    onnx_path = os.path.join(os.path.dirname(model_dir), "tmp.onnx")
    if not os.path.exists(engine_path) or force:
        export_onnx(model, image, onnx_path)
        export_engine(image.shape[2],
                      image.shape[3],
                      max_batch_size=max_batch_size,
                      onnx_path=onnx_path,
                      engine_path=engine_path,
                      dtype=dtype)

    runs = 5
    print(">>>> trt")
    trt_session = TrtSession(engine_path=engine_path, dtype=dtype)
    for _ in range(runs):
        x1 = trt_session.infer(image)
    print("out:\n", x1[-1].cpu().numpy())

    print(">>>> pth")
    for _ in range(runs):
        x2 = model.forward(image)
    print("out:\n", x2[-1].cpu().numpy())

    print(">>>> ort")
    ort_sess = OnnxSession(onnx_path)
    for _ in range(runs):
        x3 = ort_sess.forward(image)
    print("out:\n", x3[-1])

    print(
        "pth vs trt cosine: ",
        torch.cosine_similarity(x1[-1].cpu().reshape(1, -1),
                                x2[-1].cpu().reshape(1, -1)))
    print(
        "pth vs onnx cosine: ",
        torch.cosine_similarity(x1[-1].cpu().reshape(1, -1),
                                torch.tensor(x3[-1]).reshape(1, -1)))
    print(
        "trt vs onnx cosine: ",
        torch.cosine_similarity(x2[-1].cpu().reshape(1, -1),
                                torch.tensor(x3[-1]).reshape(1, -1)))
    print("out shape: ", x2.shape)
