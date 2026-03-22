import torch
from models.conv3dCNN import ImprovedSkeletonNet
from models.action_format import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class conv3DCNN_model:
    def __init__(self, params, threshold):
        self.weights = params["weights"]
        self.threshold = threshold
        self.model=ImprovedSkeletonNet(num_classes=params['num_classes'],num_people=params["bodies"]).to(DEVICE)
        self.model.load_weights(params["weights"])

    def predict(self,poses):
        # Принимает список объектов Pose
        result = self.model.predict(poses)
        action = Action()
        if result['confidence'] > self.threshold:
            action.action_id =result['predicted_class']
            action.action_name =ACTIONS[result['predicted_class']]
            action.conf =result['confidence']
        return action,result