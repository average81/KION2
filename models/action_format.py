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
    60: 'side lunge', 61: 'abseiling', 62: 'air drumming', 63: 'answering questions', 64: 'applauding',
    65: 'applying cream', 66: 'archery', 67: 'arm wrestling', 68: 'arranging flowers', 69: 'assembling computer',
    70: 'auctioning', 71: 'baby waking up', 72: 'baking cookies', 73: 'balloon blowing', 74: 'bandaging',
    75: 'barbequing', 76: 'bartending', 77: 'beatboxing', 78: 'bee keeping', 79: 'belly dancing',
    80: 'bench pressing', 81: 'bending back', 82: 'bending metal', 83: 'biking through snow',
    84: 'blasting sand', 85: 'blowing glass', 86: 'blowing leaves', 87: 'blowing nose',
    88: 'blowing out candles', 89: 'bobsledding', 90: 'bookbinding', 91: 'bouncing on trampoline',
    92: 'bowling', 93: 'braiding hair', 94: 'breading or breadcrumbing', 95: 'breakdancing',
    96: 'brush painting', 97: 'building cabinet', 98: 'building shed', 99: 'bungee jumping',
    100: 'busking', 101: 'canoeing or kayaking', 102: 'capoeira', 103: 'carrying baby',
    104: 'cartwheeling', 105: 'carving pumpkin', 106: 'catching fish', 107: 'catching or throwing baseball',
    108: 'catching or throwing frisbee', 109: 'catching or throwing softball', 110: 'celebrating',
    111: 'changing oil', 112: 'changing wheel', 113: 'checking tires', 114: 'cheerleading',
    115: 'chopping wood', 116: 'clay pottery making', 117: 'clean and jerk', 118: 'cleaning floor',
    119: 'cleaning gutters', 120: 'cleaning pool', 121: 'cleaning shoes', 122: 'cleaning toilet',
    123: 'cleaning windows', 124: 'climbing a rope', 125: 'climbing ladder', 126: 'climbing tree',
    127: 'contact juggling', 128: 'cooking chicken', 129: 'cooking egg', 130: 'cooking on campfire',
    131: 'cooking sausages', 132: 'counting money', 133: 'country line dancing', 134: 'cracking neck',
    135: 'crawling baby', 136: 'crossing river', 137: 'crying', 138: 'curling hair',
    139: 'cutting nails', 140: 'cutting pineapple', 141: 'cutting watermelon', 142: 'dancing ballet',
    143: 'dancing charleston', 144: 'dancing gangnam style', 145: 'dancing macarena', 146: 'deadlifting',
    147: 'decorating the christmas tree', 148: 'digging', 149: 'dining', 150: 'disc golfing',
    151: 'diving cliff', 152: 'dodgeball', 153: 'doing aerobics', 154: 'doing laundry',
    155: 'doing nails', 156: 'drawing', 157: 'dribbling basketball', 158: 'drinking beer',
    159: 'drinking shots', 160: 'driving car', 161: 'driving tractor', 162: 'drop kicking',
    163: 'drumming fingers', 164: 'dunking basketball', 165: 'dying hair', 166: 'eating burger',
    167: 'eating cake', 168: 'eating carrots', 169: 'eating chips', 170: 'eating doughnuts',
    171: 'eating hotdog', 172: 'eating ice cream', 173: 'eating spaghetti', 174: 'eating watermelon',
    175: 'egg hunting', 176: 'exercising arm', 177: 'exercising with an exercise ball', 178: 'extinguishing fire',
    179: 'faceplanting', 180: 'feeding birds', 181: 'feeding fish', 182: 'feeding goats',
    183: 'filling eyebrows', 184: 'finger snapping', 185: 'fixing hair', 186: 'flipping pancake',
    187: 'flying kite', 188: 'folding clothes', 189: 'folding napkins', 190: 'folding paper',
    191: 'front raises', 192: 'frying vegetables', 193: 'garbage collecting', 194: 'gargling',
    195: 'getting a haircut', 196: 'getting a tattoo', 197: 'giving or receiving award',
    198: 'golf chipping', 199: 'golf driving', 200: 'golf putting', 201: 'grinding meat',
    202: 'grooming dog', 203: 'grooming horse', 204: 'gymnastics tumbling', 205: 'hammer throw',
    206: 'headbanging', 207: 'headbutting', 208: 'high jump', 209: 'high kick', 210: 'hitting baseball',
    211: 'hockey stop', 212: 'holding snake', 213: 'hopscotch', 214: 'hoverboarding', 215: 'hugging',
    216: 'hula hooping', 217: 'hurdling', 218: 'hurling (sport)', 219: 'ice climbing',
    220: 'ice fishing', 221: 'ice skating', 222: 'ironing', 223: 'javelin throw', 224: 'jetskiing',
    225: 'jogging', 226: 'juggling balls', 227: 'juggling fire', 228: 'juggling soccer ball',
    229: 'jumping into pool', 230: 'jumpstyle dancing', 231: 'kicking field goal', 232: 'kicking soccer ball',
    233: 'kissing', 234: 'kitesurfing', 235: 'knitting', 236: 'krumping', 237: 'laughing',
    238: 'laying bricks', 239: 'long jump', 240: 'making a cake', 241: 'making a sandwich',
    242: 'making bed', 243: 'making jewelry', 244: 'making pizza', 245: 'making snowman',
    246: 'making sushi', 247: 'making tea', 248: 'marching', 249: 'massaging back',
    250: 'massaging feet', 251: 'massaging legs', 252: 'massaging person\'s head', 253: 'milking cow',
    254: 'mopping floor', 255: "motorcycling",255: 'moving furniture', 256: 'mowing lawn', 257: 'news anchoring',
    258: 'opening bottle', 259: 'opening present', 260: 'paragliding', 261: 'parasailing',
    262: 'parkour', 263: 'passing American football (in game)', 264: 'passing American football (not in game)',
    265: 'peeling apples', 266: 'peeling potatoes', 267: 'petting animal (not cat)', 268: 'petting cat',
    269: 'picking fruit', 270: 'planting trees', 271: 'plastering', 272: 'playing accordion',
    273: 'playing badminton', 274: 'playing bagpipes', 275: "playing basketball",275: 'playing cards', 276: 'playing cello',
    277: 'playing chess', 278: 'playing clarinet', 279: 'playing controller', 280: 'playing cricket',
    281: 'playing cymbals', 282: 'playing didgeridoo', 283: 'playing drums', 284: 'playing flute',
    285: 'playing guitar', 286: 'playing harmonica', 287: 'playing harp', 288: 'playing ice hockey',
    289: 'playing keyboard', 290: 'playing kickball', 291: 'playing monopoly', 292: 'playing organ',
    293: 'playing paintball', 294: 'playing piano', 295: 'playing poker', 296: 'playing recorder',
    297: 'playing saxophone', 298: 'playing squash or racquetball', 299: 'playing tennis',
    300: 'playing trombone', 301: 'playing trumpet', 302: 'playing ukulele', 303: 'playing violin',
    304: 'playing volleyball', 305: 'playing xylophone', 306: 'pole vault', 307: 'presenting weather forecast',
    308: 'pull ups', 309: 'pumping fist', 310: 'pumping gas', 311: 'punching bag', 312: 'punching person (boxing)',
    313: 'push up', 314: 'pushing car', 315: 'pushing cart', 316: 'pushing wheelchair', 317: 'reading book',
    318: 'reading newspaper', 319: 'recording music', 320: 'riding a bike', 321: 'riding camel',
    322: 'riding elephant', 323: 'riding mechanical bull', 324: 'riding mountain bike', 325: 'riding mule',
    326: 'riding or walking with horse', 327: 'riding scooter', 328: 'riding unicycle', 329: 'ripping paper',
    330: 'robot dancing', 331: 'rock climbing', 332: 'rock scissors paper', 333: 'roller skating',
    334: 'running on treadmill', 335: 'sailing', 336: 'salsa dancing', 337: 'sanding floor',
    338: 'scrambling eggs', 339: 'scuba diving', 340: 'setting table', 341: 'shaking hands',
    342: 'shaking head', 343: 'sharpening knives', 344: 'sharpening pencil', 345: 'shaving head',
    346: 'shaving legs', 347: 'shearing sheep', 348: 'shining shoes', 349: 'shooting basketball',
    350: 'shooting goal (soccer)', 351: 'shot put', 352: 'shoveling snow', 353: 'shredding paper',
    354: 'shuffling cards', 355: 'sign language interpreting', 356: 'singing', 357: 'situp',
    358: 'skateboarding', 359: 'ski jumping', 360: 'skiing (not slalom or crosscountry)',
    361: 'skiing crosscountry', 362: 'skiing slalom', 363: 'skipping rope', 364: 'skydiving',
    365: 'slacklining', 366: 'slapping', 367: 'sled dog racing', 368: 'smoking', 369: 'smoking hookah',
    370: 'snatch weight lifting', 371: 'sneezing', 372: 'sniffing', 373: 'snorkeling',
    374: 'snowboarding', 375: 'snowkiting', 376: 'snowmobiling', 377: 'somersaulting',
    378: 'spinning poi', 379: 'spray painting', 380: 'spraying', 381: 'springboard diving',
    382: 'sticking tongue out', 383: 'stomping grapes', 384: 'stretching leg', 385: 'strumming guitar',
    386: 'surfing crowd', 387: 'surfing water', 388: 'sweeping floor', 389: 'swimming backstroke',
    390: 'swimming breast stroke', 391: 'swimming butterfly stroke', 392: 'swing dancing',
    393: 'swinging legs', 394: 'swinging on something', 395: 'sword fighting', 396: 'tai chi',
    397: 'taking a shower', 398: 'tango dancing', 399: 'tap dancing', 400: 'tapping guitar',
    401: 'tapping pen', 402: 'tasting beer', 403: 'tasting food', 404: 'testifying', 405: 'texting',
    406: 'throwing axe', 407: 'throwing ball', 408: 'throwing discus', 409: 'tickling',
    410: 'tobogganing', 411: 'tossing coin', 412: 'tossing salad', 413: 'training dog',
    414: 'trapezing', 415: 'trimming or shaving beard', 416: 'trimming trees', 417: 'triple jump',
    418: 'tying bow tie', 419: 'tying knot (not on a tie)', 420: 'tying tie', 421: 'unboxing',
    422: 'unloading truck', 423: 'using computer', 424: 'using remote controller (not gaming)',
    425: 'using segway', 426: 'vault', 427: 'waiting in line', 428: 'walking the dog',
    429: 'washing dishes', 430: 'washing feet', 431: 'washing hair', 432: 'washing hands',
    433: 'water skiing', 434: 'water sliding', 435: 'watering plants', 436: 'waxing back',
    437: 'waxing chest', 438: 'waxing eyebrows', 439: 'waxing legs', 440: 'weaving basket',
    441: 'welding', 442: 'whistling', 443: 'windsurfing', 444: 'wrapping present', 445: 'wrestling',
    446: 'yawning', 447: 'yoga', 448: 'zumba'
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
