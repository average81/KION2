import torch
from typing import List, Dict, Any
import numpy as np
from models.stgcn.stgcn_wrapper import STGCNWrapper  # Используем обертку вместо прямого доступа к STGCN
from models.action_format import *

#Сопоставление индексов модели индексам объекта Action
STGN_ACTIONS_MAPPING = {
    0: 61,
    1: 163,
    2: 63,
    3: 9,
    4: 65,
    5: 66,
    6: 67,
    7: 68,
    8: 69,
    9: 70,
    10: 71,
    11: 72,
    12: 73,
    13: 74,
    14: 75,
    15: 76,
    16: 77,
    17: 78,
    18: 79,
    19: 45,
    20: 81,
    21: 82,
    22: 83,
    23: 84,
    24: 85,
    25: 86,
    26: 87,
    27: 88,
    28: 89,
    29: 90,
    30: 91,
    31: 92,
    32: 93,
    33: 94,
    34: 95,
    35: 96,
    36: 3,
    37: 2,
    38: 97,
    39: 98,
    40: 99,
    41: 100,
    42: 101,
    43: 102,
    44: 103,
    45: 104,
    46: 105,
    47: 106,
    48: 107,
    49: 108,
    50: 109,
    51: 110,
    52: 111,
    53: 112,
    54: 113,
    55: 114,
    56: 115,
    57: 9,
    58: 116,
    59: 117,
    60: 118,
    61: 119,
    62: 120,
    63: 121,
    64: 122,
    65: 123,
    66: 124,
    67: 125,
    68: 126,
    69: 127,
    70: 128,
    71: 129,
    72: 130,
    73: 131,
    74: 132,
    75: 133,
    76: 134,
    77: 135,
    78: 136,
    79: 137,
    80: 138,
    81: 139,
    82: 140,
    83: 141,
    84: 142,
    85: 143,
    86: 144,
    87: 145,
    88: 146,
    89: 147,
    90: 148,
    91: 149,
    92: 150,
    93: 151,
    94: 152,
    95: 153,
    96: 154,
    97: 155,
    98: 156,
    99: 157,
    100: 0,
    101: 158,
    102: 159,
    103: 160,
    104: 161,
    105: 162,
    106: 163,
    107: 164,
    108: 165,
    109: 166,
    110: 167,
    111: 168,
    112: 169,
    113: 170,
    114: 171,
    115: 172,
    116: 173,
    117: 174,
    118: 175,
    119: 176,
    120: 177,
    121: 178,
    122: 179,
    123: 180,
    124: 181,
    125: 182,
    126: 183,
    127: 184,
    128: 185,
    129: 186,
    130: 187,
    131: 188,
    132: 189,
    133: 190,
    134: 191,
    135: 192,
    136: 193,
    137: 194,
    138: 195,
    139: 196,
    140: 197,
    141: 198,
    142: 199,
    143: 200,
    144: 201,
    145: 202,
    146: 203,
    147: 204,
    148: 205,
    149: 206,
    150: 207,
    151: 208,
    152: 209,
    153: 210,
    154: 211,
    155: 212,
    156: 213,
    157: 214,
    158: 215,
    159: 216,
    160: 217,
    161: 218,
    162: 219,
    163: 220,
    164: 221,
    165: 222,
    166: 223,
    167: 224,
    168: 225,
    169: 226,
    170: 227,
    171: 228,
    172: 229,
    173: 230,
    174: 231,
    175: 232,
    176: 233,
    177: 234,
    178: 235,
    179: 236,
    180: 237,
    181: 238,
    182: 239,
    183: 39,
    184: 240,
    185: 241,
    186: 242,
    187: 243,
    188: 244,
    189: 245,
    190: 246,
    191: 247,
    192: 248,
    193: 249,
    194: 250,
    195: 251,
    196: 252,
    197: 253,
    198: 254,
    199: 255,
    200: 256,
    201: 257,
    202: 258,
    203: 259,
    204: 260,
    205: 261,
    206: 262,
    207: 263,
    208: 264,
    209: 265,
    210: 266,
    211: 267,
    212: 268,
    213: 269,
    214: 270,
    215: 271,
    216: 272,
    217: 273,
    218: 274,
    219: 275,
    220: 276,
    221: 277,
    222: 278,
    223: 279,
    224: 280,
    225: 281,
    226: 282,
    227: 283,
    228: 284,
    229: 285,
    230: 286,
    231: 287,
    232: 288,
    233: 289,
    234: 290,
    235: 291,
    236: 292,
    237: 293,
    238: 294,
    239: 295,
    240: 296,
    241: 297,
    242: 298,
    243: 299,
    244: 300,
    245: 301,
    246: 302,
    247: 303,
    248: 304,
    249: 305,
    250: 306,
    251: 307,
    252: 308,
    253: 309,
    254: 310,
    255: 311,
    256: 312,
    257: 313,
    258: 314,
    259: 315,
    260: 316,
    261: 317,
    262: 318,
    263: 319,
    264: 320,
    265: 321,
    266: 322,
    267: 323,
    268: 324,
    269: 325,
    270: 326,
    271: 327,
    272: 328,
    273: 329,
    274: 330,
    275: 331,
    276: 332,
    277: 333,
    278: 334,
    279: 335,
    280: 336,
    281: 337,
    282: 338,
    283: 339,
    284: 340,
    285: 341,
    286: 342,
    287: 343,
    288: 344,
    289: 345,
    290: 346,
    291: 347,
    292: 348,
    293: 349,
    294: 350,
    295: 351,
    296: 352,
    297: 353,
    298: 354,
    299: 355,
    300: 356,
    301: 357,
    302: 56,
    303: 358,
    304: 359,
    305: 360,
    306: 361,
    307: 362,
    308: 363,
    309: 364,
    310: 365,
    311: 52,
    312: 366,
    313: 367,
    314: 368,
    315: 369,
    316: 370,
    317: 371,
    318: 372,
    319: 373,
    320: 374,
    321: 375,
    322: 376,
    323: 377,
    324: 378,
    325: 379,
    326: 380,
    327: 381,
    328: 382,
    329: 383,
    330: 40,
    331: 384,
    332: 385,
    333: 386,
    334: 387,
    335: 388,
    336: 389,
    337: 390,
    338: 391,
    339: 392,
    340: 393,
    341: 394,
    342: 395,
    343: 396,
    344: 397,
    345: 398,
    346: 399,
    347: 400,
    348: 401,
    349: 402,
    350: 403,
    351: 404,
    352: 405,
    353: 406,
    354: 407,
    355: 408,
    356: 409,
    357: 410,
    358: 411,
    359: 412,
    360: 413,
    361: 414,
    362: 415,
    363: 416,
    364: 417,
    365: 418,
    366: 419,
    367: 420,
    368: 421,
    369: 422,
    370: 423,
    371: 424,
    372: 425,
    373: 426,
    374: 427,
    375: 428,
    376: 429,
    377: 430,
    378: 431,
    379: 432,
    380: 433,
    381: 434,
    382: 435,
    383: 436,
    384: 437,
    385: 438,
    386: 439,
    387: 440,
    388: 441,
    389: 442,
    390: 443,
    391: 444,
    392: 445,
    393: 446,
    394: 447,
    395: 448,
    396: 1,
    397: 11,
    398: 399,
    399: 448
}
class STGCN_model:
    """
    Класс для работы с предобученной моделью ST-GCN через обертку.

    Поддерживает загрузку весов и предсказание действий по последовательности поз.
    """

    def __init__(self, model_params: Dict[str, Any], threshold: float = 0.5):
        """
        Инициализация модели ST-GCN через обертку.

        Args:
            model_params (dict): Параметры модели, включая:
                - weights: путь к файлу с весами (.pt)
                - label_map_path: путь к карте меток (опционально)
                - device: устройство ('cpu' или 'cuda')
            threshold (float): Порог уверенности для принятия предсказания.
        """
        self.threshold = threshold
        weights_path = model_params.get("weights")
        label_map_path = model_params.get("label_map_path", None)
        device = model_params.get("device", "cpu")

        if not weights_path:
            raise ValueError("Параметр 'weights' обязателен в model_params")

        # Создаем модель через обертку
        self.model = STGCNWrapper(
            weights_path=weights_path,
            label_map_path=label_map_path
        )

        print(f"✅ Модель ST-GCN загружена через обертку. Веса: {weights_path}")

    def predict(self, poses: List[Dict]) -> Dict[str, Any]:
        """
        Выполняет предсказание действия по последовательности поз.

        Args:
            poses (List[Dict]): Список объектов Pose или словарей с данными поз.
                                Ожидается, что каждый элемент имеет .keypoints и .id

        Returns:
            Dict: Результат с action_id, action_name и conf.
        """
        if len(poses) == 0:
            return {"action_id": -1, "action_name": "no pose", "conf": 0.0}

        # Преобразуем poses в тензор data_numpy формы (1, 3, T, V, M)
        data_numpy = self._poses_to_numpy(poses)
        #print(data_numpy)
        with torch.no_grad():
            logits = self.model.predict_logits(data_numpy)  # (1, 400)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            conf_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[conf_idx].item()
        #print(confidence)
        # Получаем название действия
        action_name = "unknown"
        if self.model.id2label is not None:
            action_name = self.model.id2label.get(str(conf_idx), f"action_{conf_idx}")
        else:
            action_name = f"kinetics_class_{conf_idx}"
        action = Action()
        if confidence > self.threshold:
            action.action_id =int(STGN_ACTIONS_MAPPING[conf_idx])
            action.action_name =action_name
            action.conf =confidence
        return action


    def _poses_to_numpy(self, poses: List[Dict]) -> np.ndarray:
        """
        Преобразует список поз в массив NumPy, совместимый с ST-GCN.

        STGCNWrapper ожидает массив формы (1, 3, T, V, M), где:
            C=3 — x, y, score
            T — количество кадров
            V=18 — количество ключевых точек (openpose layout)
            M=1 — количество людей

        Args:
            poses (List[Dict]): Список объектов Pose.

        Returns:
            np.ndarray: Массив формы (1, 3, T, V, M)
        """
        T = len(poses)
        V = 18  # openpose layout
        M = 1   # только один человек
        C = 3

        # Создаем массив данных
        data = np.zeros((C, T, V, M), dtype=np.float32)

        # Индексы из JOINTS в pose_format.py → OpenPose индексы
        JOINT_MAP = {
            25: 0,  # nose → Nose
            2: 1,   # neck → Neck
            8: 2,   # right_shoulder → RShoulder
            9: 3,   # right_elbow → RElbow
            10: 4,  # right_wrist → RWrist
            4: 5,   # left_shoulder → LShoulder
            5: 6,   # left_elbow → LElbow
            6: 7,   # left_wrist → LWrist
            16: 8,  # right_hip → RHip
            17: 9,  # right_knee → RKnee
            18: 10, # right_ankle → RAnkle
            12: 11, # left_hip → LHip
            13: 12, # left_knee → LKnee
            14: 13, # left_ankle → LAnkle
            27: 14, # right_eye → REye
            26: 15, # left_eye → LEye
            29: 16, # right_ear → REar
            28: 17  # left_ear → LEar
        }

        # Собираем все координаты для нормализации
        all_x = []
        all_y = []
        for t, pose in enumerate(poses):
            for our_idx, op_idx in JOINT_MAP.items():
                if our_idx < len(pose.keypoints):
                    x, y = pose.keypoints[our_idx]
                    score = pose.keypoints_conf[our_idx] if our_idx < len(pose.keypoints_conf) else 0.0
                    if score > 0:
                        all_x.append(x)
                        all_y.append(y)

        # Вычисляем диапазон
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            range_x = max_x - min_x if max_x != min_x else 1.0
            range_y = max_y - min_y if max_y != min_y else 1.0
        else:
            min_x = min_y = 0.0
            range_x = range_y = 1.0

        # Заполняем и нормализуем данные
        for t, pose in enumerate(poses):
            for our_idx, op_idx in JOINT_MAP.items():
                if our_idx < len(pose.keypoints):
                    x, y = pose.keypoints[our_idx]
                    score = pose.keypoints_conf[our_idx] if our_idx < len(pose.keypoints_conf) else 0.0
                    data[0, t, op_idx, 0] = (x - min_x) / range_x - 0.5  # Нормализация x
                    data[1, t, op_idx, 0] = (y - min_y) / range_y - 0.5  # Нормализация y
                    data[2, t, op_idx, 0] = score

        # Добавляем batch dimension → (1, C, T, V, M)
        return data[np.newaxis, :, :, :, :]