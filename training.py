from ultralytics import YOLO

if __name__ == '__main__': 
    model = YOLO("yolov8n.pt")  

    results = model.train(data="C:\\Users\\Jeff\Desktop\\food app\\final year proj\\training\config.yaml", epochs=12)