import sys
import argparse
from models.lstm_gcn_net import LSTMSkeletonNet, CLASSES
from app.pose_action_classificator import action_models

def main():
    parser = argparse.ArgumentParser(description='Предсказание действия с помощью LSTM-GCN модели')
    parser.add_argument('input_path', type=str, help='Путь к файлу данных (.skeleton) или директории с позами')
    parser.add_argument('--weights', type=str, default='models/best_lstm_model.pth',
                        help='Путь к весам модели')
    parser.add_argument('--config', type=str, default='configs/lstm_params.yml',
                        help='Путь к конфигурационному файлу')

    args = parser.parse_args()

    # Создаем модель
    params = action_models["LSTMSkeletonNet"]["params"]
    model = LSTMSkeletonNet(num_classes=params['num_classes'], input_size=params["input_size"],
                            hidden_size=params["hidden_size"], num_layers=params["num_layers"],
                            dropout=params["dropout"], bodies=params["bodies"])

    # Загружаем веса
    try:
        model.load_weights(args.weights)
    except Exception as e:
        print(f"❌ Не удалось загрузить веса: {e}")
        sys.exit(1)

    # Выполняем предсказание
    try:
        result = model.predict(args.input_path)

        print(f"✅ Предсказание выполнено успешно!")
        print(f"🔹 Предсказанный класс: {result['predicted_class']}, {CLASSES[result['predicted_class']]}")
        print(f"🔹 Уверенность: {result['confidence']:.3f}")
        print(f"🔹 Логиты: [{', '.join(f'{v:.3f}' for v in result['logits'][:5])}, ...]")

    except Exception as e:
        print(f"❌ Ошибка при предсказании: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()