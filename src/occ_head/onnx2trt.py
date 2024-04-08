import tensorrt as trt

TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path, engine_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network,TRT_LOGGER)
   
    with open(onnx_file_path,'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB的工作空间
    
     # 使用FP16模式如果可行
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

    return engine


if __name__ == "__main__":
    build_engine("/tmp/occ_head_opt.onnx","/tmp/occ_head.trt")