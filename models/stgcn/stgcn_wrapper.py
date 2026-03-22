# stgcn_wrapper.py
import os
import json
import torch
# Та же реализация, что и при обучении (config: model: net.st_gcn.Model)
from models.stgcn.net.st_gcn import Model

class STGCNWrapper:
    def __init__(
        self,
        weights_path='./models/st_gcn.kinetics.pt',
        label_map_path='kinetics400-id2label.txt',
        device='cpu',
        num_class: int = 400,
    ):
        self.device = torch.device(device)

        # модель как в config/st_gcn/kinetics-skeleton/test.yaml
        self.model = Model(
            in_channels=3,
            num_class=num_class,
            edge_importance_weighting=True,
            graph_args={'layout': 'openpose', 'strategy': 'spatial'}
        ).to(self.device)

        ckpt = torch.load(weights_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.model.load_state_dict(ckpt)
        self.model.eval()

        # загрузка словаря id->label (можно отключить, если не нужен)
        self.id2label = None
        if label_map_path is not None and os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.id2label = json.load(f)

    @torch.no_grad()
    def predict_logits(self, data_numpy):
        """
        data_numpy: np.ndarray формы (1, 3, T, V, M), V=18, M=1
        возвращает torch.Tensor формы (1, num_class)
        """
        data_tensor = torch.from_numpy(data_numpy).float().to(self.device)
        out = self.model(data_tensor)
        return out

    @torch.no_grad()
    def predict_topk(self, data_numpy, k=5):
        """
        Возвращает список top-k (cls_id, prob, label_str)
        """
        logits = self.predict_logits(data_numpy)  # (1,num_class)
        prob = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(prob, k=k)

        results = []
        for cls_id, p in zip(topk.indices.tolist(), topk.values.tolist()):
            label = None
            if self.id2label is not None:
                label = self.id2label.get(str(cls_id), None)
            results.append((cls_id, float(p), label))
        return results