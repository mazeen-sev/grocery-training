from ultralytics import YOLO

if __name__ == '__main__': 
    model = YOLO('runs/detect/train5/weights/best.pt')

    results = model.val('https://m.media-amazon.com/images/I/81HOHSv1gjL._AC_UF1000,1000_QL80_.jpg')

    