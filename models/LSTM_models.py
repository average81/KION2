import torch
from models.lstm_gcn_net import LSTMSkeletonNet
from models.action_format import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LSTM_model:
    def __init__(self, params, threshold):
        self.weights = params["weights"]
        self.threshold = threshold
        self.model=LSTMSkeletonNet(num_classes=params['num_classes'], input_size=params["input_size"],
                                   hidden_size=params["hidden_size"], num_layers=params["num_layers"],
                                   dropout=params["dropout"], bodies=params["bodies"]).to(DEVICE)
        self.model.load_weights(params["weights"])

    def predict(self,poses):
        # Принимает список объектов Pose
        poses = poses[:30]
        result = self.model.predict(poses)
        action = Action()
        if result['confidence'] > self.threshold:
            action.action_id =result['predicted_class']
            action.action_name =ACTIONS[result['predicted_class']]
            action.conf =result['confidence']
        return action