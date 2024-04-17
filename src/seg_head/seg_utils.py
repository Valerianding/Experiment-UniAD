import torch
def IOU(intputs, targets):
    numerator = (intputs * targets).sum(dim=1)
    denominator = intputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss.cpu(), numerator.cpu(), denominator.cpu()

def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)      
