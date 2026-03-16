# %%
# ==============================================================================
# 1. ИМПОРТЫ (исправленный datetime)
# ==============================================================================
import os
from models.lstm_gcn_net import *
from app.pose_action_classificator import action_models


# %%
# ==============================================================================
# 5. ЗАПУСК
# ==============================================================================
if __name__ == "__main__":
    # Создаем директории
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # Пример использования
    params = action_models["LSTMSkeletonNet"]["params"]
    model = LSTMSkeletonNet(num_classes=params['num_classes'],
                            hidden_size=params["hidden_size"], num_layers=params["num_layers"],
                            dropout=params["dropout"], bodies=params["bodies"])
    #model = LSTMSkeletonNet(num_classes=60)
    model.train_model('lstm_params.yml')