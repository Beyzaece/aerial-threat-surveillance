from ultralytics import RTDETR


def main():

    
    model = RTDETR("rtdetr-l.pt")

    
    model.train(
        data="dataset.yaml",   
        epochs=30,             
        imgsz=640,            
        batch=4,               
        device=0,          
        workers=0,            
        project="runs",
        name="rtdetr_training"
    )


if __name__ == "__main__":
    main()