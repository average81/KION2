ACTIONS = {
    0: 'drink water', 1: 'eat meal', 2: 'brushing teeth', 3: 'brushing hair', 4: 'drop',
    5: 'pickup', 6: 'throw', 7: 'sitting down', 8: 'standing up', 9: 'clapping', 10: 'reading',
    11: 'writing', 12: 'tear up paper', 13: 'wear jacket', 14: 'taking off jacket',
    15: 'wear a shoe', 16: 'taking off a shoe', 17: 'wear socks', 18: 'taking off socks',
    19: 'stretching arm', 20: 'kicking', 21: 'punching', 22: 'kicking 2', 23: 'punching 2',
    24: 'falling', 25: 'hammering', 26: 'kicking something', 27: 'punching 3', 28: 'dancing',
    29: 'kicking 3', 30: 'writing 2', 31: 'taking a selfie', 32: 'checking time',
    33: 'rub two hands together', 34: 'walking zigzag', 35: 'walking with irregular speed',
    36: 'walking with heavy steps', 37: 'arm circles', 38: 'arm swings', 39: 'lunge',
    40: 'squats', 41: 'banded squats', 42: 'arm curls', 43: 'prior box squats', 44: 'pushups',
    45: 'bench press', 46: 'deadlift', 47: 'jump jacks', 48: 'rowing', 49: 'running on treadmill',
    50: 'situps', 51: 'lunges', 52: 'jump rope', 53: 'pushup jacks', 54: 'high knees',
    55: 'heels down', 56: 'side kick', 57: 'round house kick', 58: 'fore kick', 59: 'side kick 2',
    60: 'side lunge'
}

class Action:
    def __init__(self):
        self.action_id = -1
        self.action_name = ''
        self.conf = 0
    def to_dict(self):
        """
        Преобразует объект Action в словарь для сериализации в JSON.

        Returns:
            dict: Сериализованное представление объекта Action.
        """
        return {
            "action_id": int(self.action_id),
            "action_name": str(self.action_name),
            "conf": float(self.conf)
        }

    @classmethod
    def from_dict(cls, data):
        """
        Восстанавливает объект Action из словаря (например, загруженного из JSON).

        Args:
            data (dict): Данные для восстановления объекта Action.

        Returns:
            Action: Восстановленный объект Action.
        """
        action = cls()
        action.action_id = int(data.get("action_id", -1))
        action.action_name = str(data.get("action_name", ""))
        action.conf = float(data.get("conf", 0.0))
        return action