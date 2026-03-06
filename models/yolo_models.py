from ultralytics import YOLO

class YoloModel:
    def __init__(self, params, threshold):
        self.weights = params["weights"]
        self.threshold = threshold
        self.model=YOLO(self.weights)
    def detect(self,image):
        return self.model(image, conf=self.threshold, verbose=False)