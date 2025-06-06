import torch
from MobileSAM.mobile_sam.build_sam import build_sam_vit_t
import cv2
import numpy as np

def data_loader(img_path, label_path, model_input_size):

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)          # [H, W, 3], BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # Convert to RGB


    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)    # [H, W], single channel
    if label.ndim == 3:
        label = label[..., 0]                               # If accidentally 3D, take first channel

    original_size = image.shape[:2]                         # (H, W)

    image_resized = cv2.resize(image, (model_input_size, model_input_size), interpolation=cv2.INTER_LINEAR)

    # Convert to float and normalize
    pixel_mean = np.array([123.675, 116.28, 103.53])
    pixel_std = np.array([58.395, 57.12, 57.375])
    image_norm = (image_resized.astype(np.float32) - pixel_mean) / pixel_std

    # Convert to torch tensor [3, H, W]
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()

    return {"image": image_tensor,
            "label": label,
            "original_size": original_size}

def prompt_generator(image, mask, prompt_type, original_size, model_input_size):
    assert prompt_type in {"point", "box"}, "Input must be 'point' or 'box'"

    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No foreground found!")
    
    h_scale = model_input_size / original_size[0]
    w_scale = model_input_size / original_size[1]

    if prompt_type == "point":
        idx = np.random.randint(len(xs))
        point_coord = np.array([[xs[idx], ys[idx]]], dtype=np.float32)  # [[x, y]]
        point_label = np.array([1], dtype=np.int64)                     # 1 = foreground
        point_coord_resized = np.array([[point_coord[0,0] * w_scale, point_coord[0,1] * h_scale]], dtype=np.float32)
        batched_input = [{"image": image,  # [3, H, W], already normalized/resized
                          "original_size": original_size,  # (H, W)
                          "point_coords": torch.from_numpy(point_coord_resized).unsqueeze(0),  # [1, N, 2]
                          "point_labels": torch.from_numpy(point_label).unsqueeze(0),          # [1, N]
                          }]
    elif prompt_type == "box":
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min * w_scale, y_min * h_scale,
                        x_max * w_scale, y_max * h_scale], dtype=np.float32)  # [x0, y0, x1, y1]
        box_tensor = torch.from_numpy(box).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]

        batched_input = [{"image": image,  # [3, H, W]
                          "original_size": original_size,  # (H, W)
                          "boxes": box_tensor.float(),     # [1, 1, 4]
                          }]

    return batched_input


def model(prompt, checkpoint):
    sam_model = build_sam_vit_t(checkpoint=checkpoint)
    sam_model.eval()
    with torch.no_grad():
        outputs = sam_model(prompt, multimask_output=False)
        pred_mask = outputs[0]['masks'][0, 0]  # [H, W] binary mask
        pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
    return pred_mask

if __name__ == "__main__":

    data = data_loader(img_path, label_path, model_input_size)
    prompt = prompt_generator(data["image"], data["mask"], prompt_type, data["original_size"], model_input_size)
    pred = model(prompt, checkpoint)
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)