import io
import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


from pdf2image import convert_from_bytes, convert_from_path

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

def convert(size, box):

    dh = 1./(size[0])
    dw = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def output_yolo(img_path, img, rects):

    f = open(img_path+".txt", "w")
    for _, rows in rects.iterrows():
        xmin = rows["xmin"]
        ymin = rows["ymin"]
        xmax = rows["xmax"] 
        ymax = rows["ymax"] 

        x, y, w, h = convert((img.shape[0], img.shape[1]), (xmin, xmax, ymin, ymax))   
        f.write(str(0) + " " + str(x) + " " + str(y) +" " + str(w) +" " + str(h) + '\n')

def draw_detections(boxes, img):

    bboxes = boxes[0]['boxes_xyxy']
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
        

    return img

def read_img(pdf_file):

    # convert pdf to image 
    images = convert_from_bytes(pdf_file)

    # *********** take 1st pdf page only********************************************************************************************************************************************************************************************************
    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR) 
    image = image[..., ::-1]

    return image


def init():
    global model
    global output_path

    try:
        if local:
            model_path= "./../../../yolo/best3.pt"
            output_path= "."
    except:
        model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "best")
        # *********** output path feature not available for online endpoint ?!********************************************************************************************************************************************************************
        #output_path = os.environ["AZUREML_BI_OUTPUT_PATH"]

    # undo the below 2 comments when testing the inference.py file locally

    #model_path= "./../../../yolo/best3.pt"
    #output_path= "."

    # when testing the inference.py file during deployment

    #model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "best")
    #output_path = os.environ["AZUREML_BI_OUTPUT_PATH"]

    # load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = 0.25  # NMS confidence threshold
    model.eval()



def run(img):
    
    # preprocess the data
    img = read_img(img)

    # perform inference
    results = model(img, size=1280)
    det = parse_detections(results)
    img = draw_detections(det, img)

    # save the image with bbox drawn
    img_name = "temp"
    image_path = output_path + "/" + img_name + ".jpg"
    cv2.imwrite(image_path, img)
    
    # save the coordinates in yolo format
    coords=results.pandas().xyxy[0]
    output_yolo(output_path + "/" + img_name, img, coords)

    # return number of PCBs detected
    return str(len(coords["xmin"]))
    
def main():

    global local
    local = True
    # pdf input for local testing
    pdf_file = ["./../../../Inference/pdfs/-a_AN00985602_Panel-Drawing_8er_without the middle.pdf", "./../../../Inference/pdfs/-b_AN00985600_Panel-Drawing_4er.pdf"]
    init()
    f = open(pdf_file[0], 'rb')
    print(run(f.read()))
    
        
# execute main function if this file is run as a script
if __name__ == '__main__':
    main()  
    

