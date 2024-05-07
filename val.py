import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/visdrone_ablation/Ours/weights/best.pt')
    model.val(data='F:/ultralytics-main/dataset/myVisDrone.yaml',
              split='val',
              imgsz=640,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/visdrone',
              name='Ours',
              )