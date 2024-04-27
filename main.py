import os
import torch

from llava_vision_pth import LlavaVision
from llava_vision_trt import export_onnx, export_engine, TrtSession
import tempfile

if __name__ == "__main__":
    model_dir = "/data/models/llava-v1.5-7b"
    model = LlavaVision(model_dir=model_dir)
    image = model.load_image("../images", num=8)

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

    print(">>>> trt")
    for _ in range(10):
        x1 = trt_session.infer(image)
    print("out:\n", x1[-1].cpu().numpy())
    
    print(">>>> pth")
    for _ in range(10):
        x2 = model(image)
    print("out:\n", x2[-1].cpu().numpy())

    print("out shape: ", x2.shape)
    print("cosine: ", torch.cosine_similarity(x1[-1].cpu().reshape(1, -1), x2[-1].cpu().reshape(1, -1)))
