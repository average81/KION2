# %%
# ==============================================================================
# 1. ИМПОРТЫ
# ==============================================================================
import os
from models.stgcn.stgcn_wrapper import STGCNWrapper
from app.pose_action_classificator import action_models

# %%
# ==============================================================================
# 5. ЗАПУСК
# ==============================================================================
if __name__ == "__main__":
    # Создаем директории
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # Загружаем параметры модели из конфигурации
    params = action_models["STGCN_model_rgbd"]["params"]
    
    # Создаем экземпляр модели ST-GCN
    model = STGCNWrapper(
        weights_path=params.get("weights", "models/st_gcn.ntu60.pt"),
        label_map_path=params.get("label_map_path", "models/stgcn/ntu60-id2label.txt"),
        num_class=params.get("num_classes", 60)
    )
    
    # Запускаем обучение с конфигурационным файлом
    model.train_model('tests/stgcn_params.yml')