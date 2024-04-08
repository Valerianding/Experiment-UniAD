import onnx
import onnxoptimizer
from onnxsim import simplify
 
ONNX_MODEL_PATH = '/tmp/occ_head.onnx'
ONNX_SIM_MODEL_PATH = '/tmp/occ_head_opt.onnx'
 
if __name__ == "__main__":
    onnx_model = onnx.load(ONNX_MODEL_PATH)
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect!")
else:
    print("Model correct!")
    print("trying to optimize model")
    opt_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(opt_model,"/tmp/occ_head_opt.onnx")
