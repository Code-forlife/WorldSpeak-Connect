# Language dict
language_code_to_name = {
    "eng": "English",
    "cmn": "Mandarin Chinese",
    "spa": "Spanish",
    "hin": "Hindi",
    "ara": "Arabic",
    "por": "Portuguese",
    "ben": "Bengali",
    "rus": "Russian",
    "jpn": "Japanese",
    "pan": "Punjabi",
    "deu": "German",
    "jav": "Javanese",
    "msa": "Malay",
    "tel": "Telugu",
    "vie": "Vietnamese",
    "kor": "Korean",
    "fra": "French",
    "tam": "Tamil",
    "urd": "Urdu",
    "ita": "Italian",
    "tha": "Thai",
    "guj": "Gujarati",
    "pol": "Polish",
    "ukr": "Ukrainian",
    "mal": "Malayalam",
    "yor": "Yoruba",
}

LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}

# Source langs: S2ST / S2TT / ASR don't need source lang
# T2TT / T2ST use this
text_source_language_codes = [
    "eng",
    "cmn",
    "ben",
    "hin",
    "ara",
    "por",
    "rus",
    "jpn",
    "pan",
    "deu",
    "jav",
    "msa",
    "tel",
    "vie",
    "kor",
    "fra",
    "tam",
    "urd",
    "ita",
    "tha",
    "guj",
    "pol",
    "ukr",
    "mal",
    "yor",
]

TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])

# Target langs:
# S2ST / T2ST
s2st_target_language_codes =[
    "eng",
    "cmn",
    "ben",
    "hin",
    "ara",
    "por",
    "rus",
    "jpn",
    "pan",
    "deu",
    "jav",
    "msa",
    "tel",
    "vie",
    "kor",
    "fra",
    "tam",
    "urd",
    "ita",
    "tha",
    "guj",
    "pol",
    "ukr",
    "mal",
    "yor",
]

S2ST_TARGET_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in s2st_target_language_codes])

# S2TT / ASR
S2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
# T2TT
T2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES


LANG_TO_SPKR_ID = {
    "arb": [
        0
    ],
    "ben": [
        2,
        1
    ],
    "cat": [
        3
    ],
    "ces": [
        4
    ],
    "cmn": [
        5
    ],
    "cym": [
        6
    ],
    "dan": [
        7,
        8
    ],
    "deu": [
        9
    ],
    "eng": [
        10
    ],
    "est": [
        11,
        12,
        13
    ],
    "fin": [
        14
    ],
    "fra": [
        15
    ],
    "hin": [
        16
    ],
    "ind": [
        17,
        24,
        18,
        20,
        19,
        21,
        23,
        27,
        26,
        22,
        25
    ],
    "ita": [
        29,
        28
    ],
    "jpn": [
        30
    ],
    "kor": [
        31
    ],
    "mlt": [
        32,
        33,
        34
    ],
    "nld": [
        35
    ],
    "pes": [
        36
    ],
    "pol": [
        37
    ],
    "por": [
        38
    ],
    "ron": [
        39
    ],
    "rus": [
        40
    ],
    "slk": [
        41
    ],
    "spa": [
        42
    ],
    "swe": [
        43,
        45,
        44
    ],
    "swh": [
        46,
        48,
        47
    ],
    "tel": [
        49
    ],
    "tgl": [
        50
    ],
    "tha": [
        51,
        54,
        55,
        52,
        53
    ],
    "tur": [
        58,
        57,
        56
    ],
    "ukr": [
        59
    ],
    "urd": [
        60,
        61,
        62
    ],
    "uzn": [
        63,
        64,
        65
    ],
    "vie": [
        66,
        67,
        70,
        71,
        68,
        69
    ]
}