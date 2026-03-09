# import numpy as np

from app.stgcn.stgcn_wrapper import STGCNWrapper
from app.stgcn.demo_myjson import load_sequence_from_json_dir  # см. ниже
# если loader ты ещё не вынес в отдельный файл, временно импортни из старого demo_myjson

def main():
    json_dir = 'Open_pose_samples/S001C001P001R002A022_rgb.avi_output_json'

    data_numpy = load_sequence_from_json_dir(json_dir)
    print('data_numpy shape:', data_numpy.shape)

    model = STGCNWrapper(
        weights_path='models/st_gcn.kinetics.pt',
        label_map_path='app/stgcn/kinetics400-id2label.txt',
        device='cpu',
    )

    top5 = model.predict_topk(data_numpy, k=5)
    print('Top-5:')
    for cls_id, p, label in top5:
        print(cls_id, label, f'{p:.4f}')

if __name__ == '__main__':
    main()