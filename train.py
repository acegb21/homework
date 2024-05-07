import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-dyhead.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='F:/ultralytics-main/dataset/myDior.yaml',
                # cache=True,
                imgsz=640,
                epochs=100,
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='F:/ultralytics-main/runs/train/dior/ASF/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train/dior',
                name='test',
                )