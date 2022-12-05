
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.visualization import imshow_det_bboxes

class DetectorForFrames():

    def __init__(self, det_config, det_checkpoint):
        self.det_model = init_detector(det_config, det_checkpoint, device="cpu")

    def frame_with_bboxes(self, frame):
        bbox_result = inference_detector(self.det_model, frame)
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        img = imshow_det_bboxes(
            frame,
            bboxes,
            labels,
            None,
            class_names=self.det_model.CLASSES,
            score_thr=0.3,
            win_name='',
            show=False
        )

        return img