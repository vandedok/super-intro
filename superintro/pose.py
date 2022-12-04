from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector


class PoseDetector():
    
    def __init__(self, pose_config, pose_checkpoint, det_config, det_checkpoint):
        # initialize pose model
        self.pose_model = init_pose_model(pose_config, pose_checkpoint)
        # initialize detector
        self.det_model = init_detector(det_config, det_checkpoint)

    def detect_pose(self, img):
        mmdet_results = inference_detector(self.det_model, img)

        # extract person (COCO_ID=1) bounding boxes from the detection results
        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        # inference pose
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=self.pose_model.cfg.data.test.type)
        
        vis_result = vis_pose_result(
            self.pose_model,
            img,
            pose_results,
            dataset=self.pose_model.cfg.data.test.type,
            show=False
            )

        return vis_result