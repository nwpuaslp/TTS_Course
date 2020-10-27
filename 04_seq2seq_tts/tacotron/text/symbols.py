_pad = '_'
_eos = '~'
phone_set = [_pad, _eos] +  \
    ["n",   "s",   "ER",   "iong", "AE",  "HH",  "h",    "S",   "JH",   "AY",
    "W",   "DH",  "SH",    "t",   "AA",   "c",   "EY",   "j",   "ian",  "x",
    "uan", "ou",   "T",    "l",   "UH",    "D",  "e",   "sh",  "ang",  "ong",
    "in",  "iao",  "ing",  "IH",  "z",   "van", "uei", "ei",   "AW",  "i",
    "ch",  "OW",   "iang", "eng", "g",   "ve",  "K",    "M",   "P",   "ie",
    "AH",  "Z",    "q", "N",   "sil", "AO",   "Y",   "f",   "uai",  "k",
    "G",   "uo", "F",   "ZH",  "OY",   "r",   "m",   "b",    "o",   "iou",
    "zh",  "ao",  "EH",  "B",    "V",   "uang", "er",  "CH",   "d", "UW",
    "en",  "AX",  "a",    "xr",  "iii", "ua",   "TH",  "ueng", "ia", "NG",
    "R",   "v",   "an",   "L",   "u",   "ai",   "ii",  "p", "IY", "uen", "vn" ]

_phone_to_id = {p: i for i, p in enumerate(phone_set)}
_id_to_phone = {i: p for i, p in enumerate(phone_set)}

phone_to_int = {}
int_to_phone = {}
for idx, item in enumerate(phone_set):
    phone_to_int[item] = idx
    int_to_phone[idx] = item



def phone_to_sequence(phones):
    phones.append("~")
    return [_phone_to_id[p] for p in phones]

