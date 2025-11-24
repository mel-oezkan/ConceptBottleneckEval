import numpy as np
import matplotlib.pyplot as plt


def aggregate_attributions(attrib_tensor, method="absolute_mean"):
    """
    Reduce channel dimension to single 2D map for visualization.
    attrib_tensor: (B, C, H, W)
    method: "absolute_mean" (default) or "mean" or "max"
    returns: (B, H, W) numpy arrays
    """
    a = attrib_tensor.detach().cpu().numpy()
    if method == "absolute_mean":
        return np.mean(np.abs(a), axis=1)
    elif method == "mean":
        return np.mean(a, axis=1)
    elif method == "max":
        return np.max(a, axis=1)
    else:
        raise ValueError("Unknown method")

def normalize_map(attrib_map):
    """
    Normalize an attribution map (H,W) or (B,H,W) to [0,1] for visualization.
    """
    m = attrib_map.copy()
    if m.ndim == 2:
        m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    else:
        # batch
        for i in range(m.shape[0]):
            mm = m[i]
            m[i] = (mm - mm.min()) / (mm.max() - mm.min() + 1e-9)
    return m


def overlay_on_image(original_img, heatmap, alpha=0.5, cmap='jet'):
    """
    original_img: np array HxW x 3 in [0,1] or [0,255]
    heatmap: HxW normalized [0,1]
    returns: overlay rgb np array
    """
    if original_img.max() > 1.0:
        orig = original_img.astype(np.float32) / 255.0
    else:
        orig = original_img.copy()

    cmap_fn = plt.get_cmap(cmap)
    heat_rgba = cmap_fn(heatmap)

    heat_rgb = heat_rgba[..., :3]
    overlay = (1 - alpha) * orig + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)
    
    return overlay