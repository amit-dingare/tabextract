from ultralyticsplus import YOLO, render_result
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import os
import random
# disable warnings
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# load data
from datasets import load_dataset

ds = load_dataset("keremberke/table-extraction", "full")

#select the sample randomly
rand_idx = random.randint(0, len(ds['train']))
print("rand_idx:", rand_idx)

# select a sample
sample = ds['train'][rand_idx]

# load model
model = YOLO('keremberke/yolov8s-table-extraction')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image
model.overrides['device'] = 'cpu'  # 'cpu' or 'cuda'

# set image
image = sample['image']
    
# perform inference
results = model.predict(image)

# parse results
for result in results:
#result = results[0]
    boxes = result.boxes.xyxy # x1, y1, x2, y2
    print("boxes:", boxes)
    scores = result.boxes.conf
    print("scores:", scores)
    categories = result.boxes.cls
    print("categories:", categories)
    scores = result.probs # for classification models
    masks = result.masks # for segmentation models
    print("masks:", masks)
        
    # show results on image
    render = render_result(model=model, image=image, result=result)
    render.show()

# save results to disk
filename = 'data/output'+ str(rand_idx) + '.jpg'
render.save(filename)

