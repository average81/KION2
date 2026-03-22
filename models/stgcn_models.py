import torch
from typing import List, Dict, Any
import numpy as np
from models.stgcn.stgcn_wrapper import STGCNWrapper  # Используем обертку вместо прямого доступа к STGCN
from models.action_format import *

#Сопоставление индексов модели индексам объекта Action
STGN_ACTIONS_MAPPING = {
    0: 61,   # abseiling
    1: 163,  # air drumming
    2: 63,   # answering questions
    3: 9,    # applauding
    4: 65,   # applying cream
    5: 66,   # archery
    6: 67,   # arm wrestling
    7: 68,   # arranging flowers
    8: 69,   # assembling computer
    9: 70,   # auctioning
    10: 71,  # baby waking up
    11: 72,  # baking cookies
    12: 73,  # balloon blowing
    13: 74,  # bandaging
    14: 75,  # barbequing
    15: 76,  # bartending
    16: 77,  # beatboxing
    17: 78,  # bee keeping
    18: 79,  # belly dancing
    19: 80,  # bench pressing → исправлено: было 45, теперь 80
    20: 81,  # bending back
    21: 82,  # bending metal
    22: 83,  # biking through snow
    23: 84,  # blasting sand
    24: 85,  # blowing glass
    25: 86,  # blowing leaves
    26: 87,  # blowing nose
    27: 88,  # blowing out candles
    28: 89,  # bobsledding
    29: 90,  # bookbinding
    30: 91,  # bouncing on trampoline
    31: 92,  # bowling
    32: 93,  # braiding hair
    33: 94,  # breading or breadcrumbing
    34: 95,  # breakdancing
    35: 96,  # brush painting
    36: 3,   # brushing hair
    37: 2,   # brushing teeth
    38: 97,  # building cabinet
    39: 98,  # building shed
    40: 99,  # bungee jumping
    41: 100, # busking
    42: 101, # canoeing or kayaking
    43: 102, # capoeira
    44: 103, # carrying baby
    45: 104, # cartwheeling
    46: 105, # carving pumpkin
    47: 106, # catching fish
    48: 107, # catching or throwing baseball
    49: 108, # catching or throwing frisbee
    50: 109, # catching or throwing softball
    51: 110, # celebrating
    52: 111, # changing oil
    53: 112, # changing wheel
    54: 113, # checking tires
    55: 114, # cheerleading
    56: 115, # chopping wood
    57: 9,   # clapping → уже есть в ACTIONS как 9
    58: 116, # clay pottery making
    59: 117, # clean and jerk
    60: 118, # cleaning floor
    61: 119, # cleaning gutters
    62: 120, # cleaning pool
    63: 121, # cleaning shoes
    64: 122, # cleaning toilet
    65: 123, # cleaning windows
    66: 124, # climbing a rope
    67: 125, # climbing ladder
    68: 126, # climbing tree
    69: 127, # contact juggling
    70: 128, # cooking chicken
    71: 129, # cooking egg
    72: 130, # cooking on campfire
    73: 131, # cooking sausages
    74: 132, # counting money
    75: 133, # country line dancing
    76: 134, # cracking neck
    77: 135, # crawling baby
    78: 136, # crossing river
    79: 137, # crying
    80: 138, # curling hair
    81: 139, # cutting nails
    82: 140, # cutting pineapple
    83: 141, # cutting watermelon
    84: 142, # dancing ballet
    85: 143, # dancing charleston
    86: 144, # dancing gangnam style
    87: 145, # dancing macarena
    88: 146, # deadlifting → ранее было 46, теперь 146
    89: 147, # decorating the christmas tree
    90: 148, # digging
    91: 149, # dining
    92: 150, # disc golfing
    93: 151, # diving cliff
    94: 152, # dodgeball
    95: 153, # doing aerobics
    96: 154, # doing laundry
    97: 155, # doing nails
    98: 156, # drawing
    99: 157, # dribbling basketball
    100: 0,  # drinking → соответствует 'drink water'
    101: 158, # drinking beer
    102: 159, # drinking shots
    103: 160, # driving car
    104: 161, # driving tractor
    105: 162, # drop kicking
    106: 163, # drumming fingers
    107: 164, # dunking basketball
    108: 165, # dying hair
    109: 166, # eating burger
    110: 167, # eating cake
    111: 168, # eating carrots
    112: 169, # eating chips
    113: 170, # eating doughnuts
    114: 171, # eating hotdog
    115: 172, # eating ice cream
    116: 173, # eating spaghetti
    117: 174, # eating watermelon
    118: 175, # egg hunting
    119: 176, # exercising arm
    120: 177, # exercising with an exercise ball
    121: 178, # extinguishing fire
    122: 179, # faceplanting
    123: 180, # feeding birds
    124: 181, # feeding fish
    125: 182, # feeding goats
    126: 183, # filling eyebrows
    127: 184, # finger snapping
    128: 185, # fixing hair
    129: 186, # flipping pancake
    130: 187, # flying kite
    131: 188, # folding clothes
    132: 189, # folding napkins
    133: 190, # folding paper
    134: 191, # front raises
    135: 192, # frying vegetables
    136: 193, # garbage collecting
    137: 194, # gargling
    138: 195, # getting a haircut
    139: 196, # getting a tattoo
    140: 197, # giving or receiving award
    141: 198, # golf chipping
    142: 199, # golf driving
    143: 200, # golf putting
    144: 201, # grinding meat
    145: 202, # grooming dog
    146: 203, # grooming horse
    147: 204, # gymnastics tumbling
    148: 205, # hammer throw
    149: 206, # headbanging
    150: 207, # headbutting
    151: 208, # high jump
    152: 209, # high kick
    153: 210, # hitting baseball
    154: 211, # hockey stop
    155: 212, # holding snake
    156: 213, # hopscotch
    157: 214, # hoverboarding
    158: 215, # hugging
    159: 216, # hula hooping
    160: 217, # hurdling
    161: 218, # hurling (sport)
    162: 219, # ice climbing
    163: 220, # ice fishing
    164: 221, # ice skating
    165: 222, # ironing
    166: 223, # javelin throw
    167: 224, # jetskiing
    168: 225, # jogging
    169: 226, # juggling balls
    170: 227, # juggling fire
    171: 228, # juggling soccer ball
    172: 229, # jumping into pool
    173: 230, # jumpstyle dancing
    174: 231, # kicking field goal
    175: 232, # kicking soccer ball
    176: 233, # kissing
    177: 234, # kitesurfing
    178: 235, # knitting
    179: 236, # krumping
    180: 237, # laughing
    181: 238, # laying bricks
    182: 239, # long jump
    183: 39,  # lunge → было 39
    184: 240, # making a cake
    185: 241, # making a sandwich
    186: 242, # making bed
    187: 243, # making jewelry
    188: 244, # making pizza
    189: 245, # making snowman
    190: 246, # making sushi
    191: 247, # making tea
    192: 248, # marching
    193: 249, # massaging back
    194: 250, # massaging feet
    195: 251, # massaging legs
    196: 252, # massaging person's head
    197: 253, # milking cow
    198: 254, # mopping floor
    199: 255, # motorcycling
    200: 255, # moving furniture
    201: 256, # mowing lawn
    202: 257, # news anchoring
    203: 258, # opening bottle
    204: 259, # opening present
    205: 260, # paragliding
    206: 261, # parasailing
    207: 262, # parkour
    208: 263, # passing American football (in game)
    209: 264, # passing American football (not in game)
    210: 265, # peeling apples
    211: 266, # peeling potatoes
    212: 267, # petting animal (not cat)
    213: 268, # petting cat
    214: 269, # picking fruit
    215: 270, # planting trees
    216: 271, # plastering
    217: 272, # playing accordion
    218: 273, # playing badminton
    219: 274, # playing bagpipes
    220: 275, # playing basketball
    221: 285, # playing bass guitar
    222: 275, # playing cards
    223: 276, # playing cello
    224: 277, # playing chess
    225: 278, # playing clarinet
    226: 279, # playing controller
    227: 280, # playing cricket
    228: 281, # playing cymbals
    229: 282, # playing didgeridoo
    230: 283, # playing drums
    231: 284, # playing flute
    232: 285, # playing guitar
    233: 286, # playing harmonica
    234: 287, # playing harp
    235: 288, # playing ice hockey
    236: 289, # playing keyboard
    237: 290, # playing kickball
    238: 291, # playing monopoly
    239: 292, # playing organ
    240: 293, # playing paintball
    241: 294, # playing piano
    242: 295, # playing poker
    243: 296, # playing recorder
    244: 297, # playing saxophone
    245: 298, # playing squash or racquetball
    246: 299, # playing tennis
    247: 300, # playing trombone
    248: 301, # playing trumpet
    249: 302, # playing ukulele
    250: 303, # playing violin
    251: 304, # playing volleyball
    252: 305, # playing xylophone
    253: 306, # pole vault
    254: 307, # presenting weather forecast
    255: 308, # pull ups
    256: 309, # pumping fist
    257: 310, # pumping gas
    258: 311, # punching bag
    259: 312, # punching person (boxing)
    260: 313, # push up
    261: 314, # pushing car
    262: 315, # pushing cart
    263: 316, # pushing wheelchair
    264: 317, # reading book
    265: 318, # reading newspaper
    266: 319, # recording music
    267: 320, # riding a bike
    268: 321, # riding camel
    269: 322, # riding elephant
    270: 323, # riding mechanical bull
    271: 324, # riding mountain bike
    272: 325, # riding mule
    273: 326, # riding or walking with horse
    274: 327, # riding scooter
    275: 328, # riding unicycle
    276: 329, # ripping paper
    277: 330, # robot dancing
    278: 331, # rock climbing
    279: 332, # rock scissors paper
    280: 333, # roller skating
    281: 334, # running on treadmill
    282: 335, # sailing
    283: 336, # salsa dancing
    284: 337, # sanding floor
    285: 338, # scrambling eggs
    286: 339, # scuba diving
    287: 340, # setting table
    288: 341, # shaking hands
    289: 342, # shaking head
    290: 343, # sharpening knives
    291: 344, # sharpening pencil
    292: 345, # shaving head
    293: 346, # shaving legs
    294: 347, # shearing sheep
    295: 348, # shining shoes
    296: 349, # shooting basketball
    297: 350, # shooting goal (soccer)
    298: 351, # shot put
    299: 352, # shoveling snow
    300: 353, # shredding paper
    301: 354, # shuffling cards
    302: 56,  # side kick
    303: 355, # sign language interpreting
    304: 356, # singing
    305: 357, # situp
    306: 358, # skateboarding
    307: 359, # ski jumping
    308: 360, # skiing (not slalom or crosscountry)
    309: 361, # skiing crosscountry
    310: 362, # skiing slalom
    311: 363, # skipping rope
    312: 364, # skydiving
    313: 365, # slacklining
    314: 366, # slapping
    315: 367, # sled dog racing
    316: 368, # smoking
    317: 369, # smoking hookah
    318: 370, # snatch weight lifting
    319: 371, # sneezing
    320: 372, # sniffing
    321: 373, # snorkeling
    322: 374, # snowboarding
    323: 375, # snowkiting
    324: 376, # snowmobiling
    325: 377, # somersaulting
    326: 378, # spinning poi
    327: 379, # spray painting
    328: 380, # spraying
    329: 381, # springboard diving
    330: 40,  # squat
    331: 382, # sticking tongue out
    332: 383, # stomping grapes
    333: 19,  # stretching arm → ранее было 40, теперь 19
    334: 384, # stretching leg → ранее 384
    335: 385, # strumming guitar
    336: 19,  # surfing crowd
    337: 387, # surfing water
    338: 388, # sweeping floor
    339: 389, # swimming backstroke
    340: 390, # swimming breast stroke
    341: 391, # swimming butterfly stroke
    342: 392, # swing dancing
    343: 393, # swinging legs
    344: 394, # swinging on something
    345: 395, # sword fighting
    346: 396, # tai chi
    347: 397, # taking a shower
    348: 398, # tango dancing
    349: 399, # tap dancing
    350: 400, # tapping guitar
    351: 401, # tapping pen
    352: 402, # tasting beer
    353: 403, # tasting food
    354: 404, # testifying
    355: 405, # texting
    356: 406, # throwing axe
    357: 407, # throwing ball
    358: 408, # throwing discus
    359: 409, # tickling
    360: 410, # tobogganing
    361: 411, # tossing coin
    362: 412, # tossing salad
    363: 413, # training dog
    364: 414, # trapezing
    365: 415, # trimming or shaving beard
    366: 416, # trimming trees
    367: 417, # triple jump
    368: 418, # tying bow tie
    369: 419, # tying knot (not on a tie)
    370: 420, # tying tie
    371: 421, # unboxing
    372: 422, # unloading truck
    373: 423, # using computer
    374: 424, # using remote controller (not gaming)
    375: 425, # using segway
    376: 426, # vault
    377: 427, # waiting in line
    378: 428, # walking the dog
    379: 429, # washing dishes
    380: 430, # washing feet
    381: 431, # washing hair
    382: 432, # washing hands
    383: 433, # water skiing
    384: 434, # water sliding
    385: 435, # watering plants
    386: 436, # waxing back
    387: 437, # waxing chest
    388: 438, # waxing eyebrows
    389: 439, # waxing legs
    390: 440, # weaving basket
    391: 441, # welding
    392: 442, # whistling
    393: 443, # windsurfing
    394: 444, # wrapping present
    395: 445, # wrestling
    396: 446, # writing
    397: 447, # yawning
    398: 448, # yoga
    399: 1, # zumba
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
            threshold (float): Порог уверенности для принятия предсказания.
        """
        self.threshold = threshold
        weights_path = model_params.get("weights")
        label_map_path = model_params.get("label_map_path", None)
        self.actions_mapping = model_params.get("mapping", None)

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
        with torch.no_grad():
            logits = self.model.predict_logits(data_numpy)  # (1, 400)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            conf_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[conf_idx].item()
        # Получаем название действия
        if self.model.id2label is not None:
            action_name = self.model.id2label.get(str(conf_idx), f"action_{conf_idx}")
        else:
            action_name = f"class_{conf_idx}"
        action = Action()
        if confidence > self.threshold:
            if self.actions_mapping is not None:
                action.action_id =int(self.actions_mapping[conf_idx])
            else:
                action.action_id = conf_idx
            action.action_name =action_name
            action.conf =confidence

        raw_res = {
            'logits': logits,
            'probabilities': probabilities,
            'predicted_class': conf_idx,
            'confidence': confidence
        }
        return action,raw_res


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