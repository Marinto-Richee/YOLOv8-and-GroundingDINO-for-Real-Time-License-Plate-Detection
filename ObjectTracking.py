import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

videopath="los_angeles.mp4"
vi=sv.VideoInfo.from_video_path(videopath)
tracked=set()
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
    for result in model.track(source=videopath, stream=True):
        
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

        boxes__ = np.array(result.boxes.xyxy.numpy(), dtype="int")
        for box in boxes__:
            x,y,x1,y1=box
            cv2.imwrite("output/"+str(model.names[detections.class_id[-1]])+".jpg",result.orig_img[y:y1,x:x1])
        for box,cls_id in zip(result.boxes,detections.class_id):
            tracker_id=box.id.cpu().numpy().astype(int)
            x,y,x1,y1=(box.xyxy.numpy().astype(int))[0]
            img_name=str(model.names[cls_id])+"_"+str(tracker_id)+".jpg"
            path="output/"+img_name
            if not os.path.exists(path):
                cv2.imwrite("output/"+img_name,result.orig_img[y:y1,x:x1])
            else:
                img_old=cv2.imread(path)
                area_old=img_old.shape[0]*img_old.shape[1]
                area_new=(y1-y)*(x1-x)
                if area_new>area_old:
                    cv2.imwrite("output/"+img_name,result.orig_img[y:y1,x:x1])
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