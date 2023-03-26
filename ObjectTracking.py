import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
vi=sv.VideoInfo.from_video_path("Traffic_long.mp4")
LINE_START = sv.Point(0, vi.height//2+150)
LINE_END = sv.Point(vi.width, vi.height//2+150)
def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    model = YOLO("yolov8m.pt")
    for result in model.track(source="Traffic_long.mp4", stream=True, agnostic_nms=True):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[(detections.class_id == 7)|(detections.class_id == 2)|(detections.class_id == 3)]
        

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(1) == 27):
            break


if __name__ == "__main__":
    main()