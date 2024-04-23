#include "pytorch_cpp_helper.hpp"


std::string get_compiler_version();
std::string get_compiling_cuda_version();

Tensor ms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight,
                              const int im2col_step);

void ms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, const int im2col_step);

void modulated_deform_conv_forward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias);


void modulated_deform_conv_backward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);

void iou3d_nms_normal_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                              float nms_overlap_thresh);

void iou3d_nms_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                       float nms_overlap_thresh);

void iou3d_boxes_iou_bev_forward(Tensor boxes_a, Tensor boxes_b,
                                 Tensor ans_iou);                                                  
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "Forward function for ms_deform_attn",
        py::arg("value"),py::arg("spatial_shapes"), py::arg("level_start_index"), py::arg("sampling_loc"),
        py::arg("attn_weight"), py::arg("im2col_step"));
        
    m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "Backward function for ms_deform_attn", 
        py::arg("value"), py::arg("spatial_shapes"), py::arg("level_start_index"),
        py::arg("sampling_loc"), py::arg("attn_weight"), py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"), py::arg("grad_attn_weight"),
        py::arg("im2col_step"));

    m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "Forward function for modulated deformable convolution",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("ones"),
        py::arg("offset"), py::arg("mask"), py::arg("output"), py::arg("columns"),
        py::arg("kernel_h"), py::arg("kernel_w"), py::arg("stride_h"),
        py::arg("stride_w"), py::arg("pad_h"), py::arg("pad_w"), py::arg("dilation_h"),
        py::arg("dilation_w"), py::arg("group"), py::arg("deformable_group"),
        py::arg("with_bias"));

    m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "Backward function for modulated deformable convolution",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("ones"),
        py::arg("offset"), py::arg("mask"), py::arg("columns"), py::arg("grad_input"),
        py::arg("grad_weight"), py::arg("grad_bias"), py::arg("grad_offset"),
        py::arg("grad_mask"), py::arg("grad_output"), py::arg("kernel_h"),
        py::arg("kernel_w"), py::arg("stride_h"), py::arg("stride_w"),
        py::arg("pad_h"), py::arg("pad_w"), py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("group"), py::arg("deformable_group"), py::arg("with_bias"));

    m.def("iou3d_nms_normal_forward", &iou3d_nms_normal_forward, "Perform IOU 3D NMS operation",
          py::arg("boxes"), py::arg("keep"), py::arg("keep_num"), py::arg("nms_overlap_thresh"));
          
    m.def("iou3d_nms_forward", &iou3d_nms_forward, "Perform IOU 3D NMS operation",
          py::arg("boxes"), py::arg("keep"), py::arg("keep_num"), py::arg("nms_overlap_thresh"));

    m.def("iou3d_boxes_iou_bev_forward", &iou3d_boxes_iou_bev_forward, "Compute IOU BEV between two sets of boxes",
          py::arg("boxes_a"), py::arg("boxes_b"), py::arg("ans_iou"));
}
