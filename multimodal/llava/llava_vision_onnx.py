import onnxruntime as ort
import torch

from utils import calculate_time


class OnnxSession:

    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path)

    @calculate_time
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        output = self.sess.run([output_name],
                               {input_name: image.cpu().numpy()})

        return output[0]
