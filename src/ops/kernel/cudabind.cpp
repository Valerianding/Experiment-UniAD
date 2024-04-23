#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor ms_deform_attn_cuda_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

void ms_deform_attn_cuda_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

Tensor ms_deform_attn_impl_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

void ms_deform_attn_impl_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_forward, CUDA,
                     ms_deform_attn_cuda_forward);
REGISTER_DEVICE_IMPL(ms_deform_attn_impl_backward, CUDA,
                     ms_deform_attn_cuda_backward);


void modulated_deformable_im2col_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_im2col_impl(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_impl(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_impl(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);


void IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(const int num_a,
                                                   const Tensor boxes_a,
                                                   const int num_b,
                                                   const Tensor boxes_b,
                                                   Tensor ans_overlap);

void IoU3DBoxesIoUBevForwardCUDAKernelLauncher(const int num_a,
                                               const Tensor boxes_a,
                                               const int num_b,
                                               const Tensor boxes_b,
                                               Tensor ans_iou);

void IoU3DNMSForwardCUDAKernelLauncher(const Tensor boxes,
                                       unsigned long long* mask, int boxes_num,
                                       float nms_overlap_thresh);

void IoU3DNMSNormalForwardCUDAKernelLauncher(const Tensor boxes,
                                             unsigned long long* mask,
                                             int boxes_num,
                                             float nms_overlap_thresh);

void iou3d_boxes_overlap_bev_forward_cuda(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap) {
  IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(num_a, boxes_a, num_b, boxes_b,
                                                ans_overlap);
};

void iou3d_boxes_iou_bev_forward_cuda(const int num_a, const Tensor boxes_a,
                                      const int num_b, const Tensor boxes_b,
                                      Tensor ans_iou) {
  IoU3DBoxesIoUBevForwardCUDAKernelLauncher(num_a, boxes_a, num_b, boxes_b,
                                            ans_iou);
};

void iou3d_nms_forward_cuda(const Tensor boxes, unsigned long long* mask,
                            int boxes_num, float nms_overlap_thresh) {
  IoU3DNMSForwardCUDAKernelLauncher(boxes, mask, boxes_num, nms_overlap_thresh);
};

void iou3d_nms_normal_forward_cuda(const Tensor boxes, unsigned long long* mask,
                                   int boxes_num, float nms_overlap_thresh) {
  IoU3DNMSNormalForwardCUDAKernelLauncher(boxes, mask, boxes_num,
                                          nms_overlap_thresh);
};

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap);

void iou3d_boxes_iou_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                      const int num_b, const Tensor boxes_b,
                                      Tensor ans_iou);

void iou3d_nms_forward_impl(const Tensor boxes, unsigned long long* mask,
                            int boxes_num, float nms_overlap_thresh);

void iou3d_nms_normal_forward_impl(const Tensor boxes, unsigned long long* mask,
                                   int boxes_num, float nms_overlap_thresh);

REGISTER_DEVICE_IMPL(iou3d_boxes_overlap_bev_forward_impl, CUDA,
                     iou3d_boxes_overlap_bev_forward_cuda);
                     
REGISTER_DEVICE_IMPL(iou3d_boxes_iou_bev_forward_impl, CUDA,
                     iou3d_boxes_iou_bev_forward_cuda);

REGISTER_DEVICE_IMPL(iou3d_nms_forward_impl, CUDA, iou3d_nms_forward_cuda);

REGISTER_DEVICE_IMPL(iou3d_nms_normal_forward_impl, CUDA,
                     iou3d_nms_normal_forward_cuda);

REGISTER_DEVICE_IMPL(modulated_deformable_im2col_impl, CUDA,
                     modulated_deformable_im2col_cuda);

REGISTER_DEVICE_IMPL(modulated_deformable_col2im_impl, CUDA,
                     modulated_deformable_col2im_cuda);
                     
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_coord_impl, CUDA,
                     modulated_deformable_col2im_coord_cuda);