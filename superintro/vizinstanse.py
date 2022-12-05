
import yaml
import numpy as np
import matplotlib
from imgviz import color as color_module
from imgviz import draw as draw_module
from imgviz import utils


def mask_to_bbox(masks):
    bboxes = np.zeros((len(masks), 4), dtype=float)
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bbox = y1, x1, y2, x2
        bboxes[i] = bbox
    return bboxes


def mask_to_bbox(masks):
    bboxes = np.zeros((len(masks), 4), dtype=float)
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bbox = y1, x1, y2, x2
        bboxes[i] = bbox
    return bboxes


def viz_instances(
    image,
    labels,
    masks,
    scores,
    font_size=25,
    line_width=5,
    alpha=0.7,
    colormap=None,
):
    with open("../suppl/coco_classes.yml", "r") as file:
        coco_labels_to_classes = yaml.safe_load(file)

    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert masks.dtype == bool

    if image.ndim == 2:
        image = color_module.gray2rgb(image)
    assert image.ndim == 3

    assert all(label_i >= 0 for label_i in labels)

    n_instance = len(labels)

    if masks is None:
        masks = [None] * n_instance
    bboxes = mask_to_bbox(masks)
    captions = [coco_labels_to_classes[x] for x in labels]
    
    assert len(masks) == len(bboxes) == n_instance

    # colormap = label_module.label_colormap()
    colormap = matplotlib.cm.get_cmap('hsv')
    colors = colormap(np.arange(n_instance)/n_instance)[:,0:3]
    colors = (colors * 255).astype(np.uint8)

    dst = image
    image_area = image.shape[0] * image.shape[1]
    
    for instance_id in range(n_instance):
        mask = masks[instance_id]

        if mask is None or mask.sum() <= 0.:
            continue

        color_ins = colors[instance_id]

        maskviz = mask[:, :, None] * color_ins.astype(float)
        dst = dst.copy()
        dst[mask] = (1 - alpha) * image[mask].astype(float) + alpha * maskviz[
            mask
        ]


    dst = utils.numpy_to_pillow(dst)
    for instance_id in range(n_instance):
        bbox = bboxes[instance_id]
        caption = captions[instance_id] + " {:.0%}". format(scores[instance_id])
        color_cls = colors[instance_id]
        
        y1, x1, y2, x2 = bbox
        max_dim = max(y2-y1, x2-x1)
        font_coef = max_dim / np.sqrt(image_area) 
        instance_font_size= int(font_coef * font_size)
        if (y2 - y1) * (x2 - x1) == 0:
            continue

        aabb1 = np.array([y1, x1], dtype=int)
        aabb2 = np.array([y2, x2], dtype=int)
        draw_module.rectangle_(
            dst,
            aabb1,
            aabb2,
            outline=color_cls,
            width=line_width,
        )

        if caption is not None:
            for loc in ["lt", "lt"]:
                y1, x1, y2, x2 = draw_module.text_in_rectangle_aabb(
                    img_shape=(dst.height, dst.width),
                    loc=loc,
                    text=caption,
                    size=instance_font_size,
                    aabb1=aabb1,
                    aabb2=aabb2
                )
                if y1 >= 0 and x1 >= 0 and y2 < dst.height and x2 < dst.width:
                    break

            draw_module.text_in_rectangle_(
                img=dst,
                loc=loc,
                text=caption,
                size=instance_font_size,
                color=color_cls,
                background=(0,0,0),
                aabb1=aabb1,
                aabb2=aabb2
            )
    return utils.pillow_to_numpy(dst)