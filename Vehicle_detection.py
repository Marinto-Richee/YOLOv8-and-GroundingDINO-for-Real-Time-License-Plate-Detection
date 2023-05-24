import cv2
from ultralytics import YOLO
import supervision as sv
import subprocess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set the video path
videopath = "assets\\LPD.mp4"
outdir = "output/"
tracked = set()

def main():
    # Create a box annotator for bounding box visualization
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    
    # Load the YOLO model
    model = YOLO("yolov8s.pt")
    
    # Iterate over the video frames
    for result in model.track(source=videopath, stream=True, device=0):
        frame = result.orig_img
        
        # Convert YOLO detections to supervision detections
        detections = sv.Detections.from_yolov8(result)
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        # Filter detections based on class IDs
        detections = detections[
            (detections.class_id == 7)
            | (detections.class_id == 2)
            | (detections.class_id == 3)
        ]
        
        # Create labels for annotations
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for __, _, confidence, class_id, tracker_id in detections
        ]
        
        # Process each bounding box and save the cropped image if necessary
        for box, cls_id in zip(result.boxes, detections.class_id):
            if box.cls.cpu().numpy().astype(int) in [7, 2, 3]:
                if box.id is not None:
                    tracker_id = box.id.cpu().numpy().astype(int)
                    x, y, x1, y1 = (box.xyxy.numpy().astype(int))[0]
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    img_name = str(model.names[cls_id]) + "_" + str(tracker_id) + ".jpg"
                    path = os.path.join(outdir, img_name)
                    if not os.path.exists(path):
                        cv2.imwrite(path, frame[y:y1, x:x1])
                    else:
                        img_old = cv2.imread(path)
                        area_old = img_old.shape[0]
                        area_new = y1 - y
                        if area_new >= area_old:
                            cv2.imwrite(path, frame[y:y1, x:x1])
        
        # Annotate bounding boxes
        box_annotator.annotate(frame, detections)
        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )
        
        # Display the frame with annotations
        cv2.imshow("yolov8", frame)
        
        # Break loop if ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
    subprocess.run(["python", "License_Plate_detection.py"])
