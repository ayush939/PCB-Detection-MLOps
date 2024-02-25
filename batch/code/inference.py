import cv2
import os
import glob
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path


def parse_detections(results) -> List[Dict[str, List[float]]]:
    
    out = []
    for j in range(len(results.pandas().xyxy)):

        detections = results.pandas().xyxy[j]
        detections = detections.to_dict()
        boxes, confidence = [], []

        for i in range(len(detections["xmin"])):
            conf = detections["confidence"][i]
            if conf < 0.25:
                continue
            xmin = int(detections["xmin"][i])
            ymin = int(detections["ymin"][i])
            xmax = int(detections["xmax"][i])
            ymax = int(detections["ymax"][i])
            name = detections["name"][i]
            

            boxes.append((xmin, ymin, xmax, ymax))
            confidence.append(conf)

        out.append( {'boxes_xyxy': boxes,
            'cofidence_scores': confidence
            }
            )

    return out

def draw_detections(boxes, imgs):

    images = []
    for item,img in zip(boxes, imgs):
        bboxes = item['boxes_xyxy']
        for box in bboxes:
            xmin, ymin, xmax, ymax = box
            #print(img.shape)
            cv2.rectangle(
                img[..., ::-1],
                (xmin, ymin),
                (xmax, ymax),
                (255,0,0), 2)
            """
            cv2.putText(img, name, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)
            """
        images.append(img)

    return images

def init():
    global model
    global output_path
    # model path
    #model_path= "./../../../Inference/best2.pt"
    #output_path= "./"
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "best")
    output_path = os.environ["AZUREML_BI_OUTPUT_PATH"]

    # load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = 0.25  # NMS confidence threshold
      
    model.cpu()
    model.eval()

def read_img(file_path):
    
    image = cv2.imread(file_path)[..., ::-1]

    return image

# TODO: return image path (or images) and the coordinates in some format!!!
def run(mini_batch: List[str]):
    
    imgs = []
    for img in mini_batch:
        print(img)
        img = read_img(img)
        imgs.append(img) 
    
    # perform inference
    results = model(imgs, size=1280)
    det = parse_detections(results)
    imgs = draw_detections(det, imgs)

    for img in imgs:
        cv2.imwrite("image.jpg", img)
    
    output = pd.DataFrame()
    for i in range(len(mini_batch)):
        
        out=results.pandas().xyxy[i]
        img = mini_batch[i]
        img = img.split("/")[-1] 
        with open(Path(output_path) / f'{img}.npy', 'wb') as f:
            np.save(f, out.values)
        '''
        with open(Path(output_path) / f'{img}.npy', 'rb') as f:
            a = np.load(f, allow_pickle=True)
            print(a)
        #output=pd.concat([output, out], ignore_index = True)
        '''
    return mini_batch

def main():

    file_path = ["./../../../Inference/real-images/1030038005.00s00.00A.1._.30.pdf0.jpg", "./../../../Inference/real-images/1030038005.00s00.00A.1._.30.pdf0.jpg"]
    init()
    print(run(file_path))
    
        
# execute main function if this file is run as a script
if __name__ == '__main__':
    main()  
    

