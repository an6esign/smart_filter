import re


MAX_PEOPLE = 50
UNKNOWN = "unknown"


UNITS = {
    "один": 1, "одна": 1, "одно": 1, "одного": 1, "одной": 1,
    "два": 2, "две": 2, "двое": 2, "двоих": 2,
    "три": 3, "трое": 3, "троих": 3,
    "четыре": 4, "четверо": 4, "четырех": 4, "четырёх": 4, "четверых": 4,
    "пять": 5, "пятеро": 5, "пяти": 5, "пятерых": 5,
    "шесть": 6, "шестеро": 6, "шести": 6, "шестерых": 6,
    "семь": 7, "семеро": 7, "семи": 7, "семерых": 7,
    "восемь": 8, "восьмеро": 8, "восьми": 8, "восьмерых": 8,
    "девять": 9, "девятеро": 9, "девяти": 9, "девятерых": 9,
}

TEENS = {
    "десять": 10, "десятеро": 10, "десяти": 10, "десятерых": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}

TENS = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
}


WORD_NUMBERS = {}
WORD_NUMBERS.update(UNITS)
WORD_NUMBERS.update(TEENS)
WORD_NUMBERS.update(TENS)

for tens_word, tens_value in TENS.items():
    for unit_word, unit_value in UNITS.items():
        value = tens_value + unit_value

        if value <= MAX_PEOPLE:
            WORD_NUMBERS[f"{tens_word} {unit_word}"] = value


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"[^\w\s+.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def replace_word_numbers(text: str) -> str:
    for word, number in sorted(
        WORD_NUMBERS.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):
        text = re.sub(
            rf"\b{re.escape(word)}\b",
            str(number),
            text
        )

    return text


def valid_people_number(number: int) -> bool:
    return 1 <= number <= MAX_PEOPLE


def is_age_context(text: str, start: int, end: int) -> bool:
    window_after = text[end:end + 20]
    window_before = text[max(0, start - 18):start]

    age_after_patterns = [
        r"\s*лет\b",
        r"\s*года\b",
        r"\s*год\b",
        r"\s*годиков\b",
        r"\s*\+",
        r"\s*-\s*\d{1,2}\s*лет\b",
    ]

    age_before_patterns = [
        r"возраст\s*$",
        r"возрастом\s*$",
        r"по\s*$",
        r"ему\s*$",
        r"ей\s*$",
        r"им\s*$",
        r"одному\s*$",
        r"одной\s*$",
        r"одному из них\s*$",
        r"всем\s+по\s*$",
    ]

    for pattern in age_after_patterns:
        if re.match(pattern, window_after):
            return True

    for pattern in age_before_patterns:
        if re.search(pattern, window_before):
            return True

    return False


def extract_number_of_people(text: str, has_number: int):
    try:
        has_number = int(has_number)
    except Exception:
        has_number = 0

    if has_number != 1:
        return UNKNOWN

    text = normalize_text(text)
    text = replace_word_numbers(text)

    # 1. "от 4 до 7" -> 7
    match = re.search(
        r"\bот\s+(\d{1,2})\s+до\s+(\d{1,2})\b",
        text
    )

    if match:
        number = int(match.group(2))

        if valid_people_number(number) and not is_age_context(
            text,
            match.start(2),
            match.end(2)
        ):
            return number

    # 2. Диапазоны игроков: "5-6 человек" -> 6
    range_patterns = [
        r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)?\b",
        r"\bна\s+(\d{1,2})\s*[-–]\s*(\d{1,2})\b",
        r"\bнас\s+(\d{1,2})\s*[-–]\s*(\d{1,2})\b",
    ]

    for pattern in range_patterns:
        for match in re.finditer(pattern, text):
            number = int(match.group(2))

            if valid_people_number(number) and not is_age_context(
                text,
                match.start(2),
                match.end(2)
            ):
                return number

    # 3. Сначала ищем явное общее количество:
    # "нас будет 6 человек", "всего 6 человек", "будет 6 человек"
    #
    # Это нужно, чтобы:
    # "нас будет 6 человек, 2 взрослых и 4 ребенка" -> 6
    # а не 6 + 2 + 4 = 12
    total_patterns = [
        r"\bнас\s+(?:будет\s+|будем\s+)?(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",
        r"\bвсего\s+(?:будет\s+)?(?:нас\s+)?(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",
        r"\bбудет\s+(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",
        r"\bбудем\s+(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",
        r"\bмы\s+(?:будем\s+)?(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",
        r"\bдля\s+(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",
        r"\bна\s+(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",
    ]

    for pattern in total_patterns:
        for match in re.finditer(pattern, text):
            number = int(match.group(1))

            if valid_people_number(number) and not is_age_context(
                text,
                match.start(1),
                match.end(1)
            ):
                return number

    # 4. Составные группы:
    # "2 взрослых и 3 детей" -> 5
    # "2 парня 3 девушки" -> 5
    # "3 ребенка" -> 3
    #
    # Здесь специально НЕ используем:
    # "человек", "чел", "игроков", "участников"
    #
    # Потому что это чаще общее количество, а не состав группы.
    group_pattern = re.compile(
        r"\b(\d{1,2})\s*"
        r"(взрослых|взрослые|взрослый|взрослого|взрослая|"
        r"детей|дети|ребенка|ребенок|ребёнка|ребёнок|"
        r"ребят|ребята|"
        r"подростков|подростка|подросток|подростки|"
        r"школьников|школьника|школьник|школьники|"
        r"мальчиков|мальчика|мальчик|мальчики|"
        r"девочек|девочки|девочка|"
        r"парней|парня|парень|парни|"
        r"девушек|девушки|девушка|"
        r"мужчин|мужчины|мужчина|"
        r"женщин|женщины|женщина|"
        r"родителей|родителя|родитель|"
        r"мамы|мам|мама|"
        r"папы|пап|папа)\b"
    )

    group_numbers = []

    for match in group_pattern.finditer(text):
        number = int(match.group(1))

        if valid_people_number(number) and not is_age_context(
            text,
            match.start(1),
            match.end(1)
        ):
            group_numbers.append(number)

    if len(group_numbers) >= 2:
        total = sum(group_numbers)

        if valid_people_number(total):
            return total

    if len(group_numbers) == 1:
        return group_numbers[0]

    # 5. Явные паттерны количества
    patterns = [
        r"\bнас\s+(?:будет\s+|будем\s+)?(\d{1,2})\b",
        r"\bмы\s+(?:будем\s+)?(\d{1,2})\b",
        r"\bбудем\s+(\d{1,2})\b",
        r"\bбудет\s+(\d{1,2})\b",

        r"\bдля\s+(\d{1,2})\b",
        r"\bна\s+(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)?\b",
        r"\bдля\s+(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)?\b",

        r"\b(\d{1,2})\s*(человек|человека|чел|игроков|игрока|участников|персон|ребят|ребята)\b",

        r"\bкомпания\s+(?:из\s+)?(\d{1,2})\b",
        r"\bгруппа\s+(?:из\s+)?(\d{1,2})\b",
        r"\bкоманд[ауеы]\s+(?:из\s+)?(\d{1,2})\b",

        r"\bчеловек\s+(\d{1,2})\b",
        r"\bчеловека\s+(\d{1,2})\b",
        r"\bчел\s+(\d{1,2})\b",
        r"\bигроков\s+(\d{1,2})\b",
        r"\bучастников\s+(\d{1,2})\b",
        r"\bребят\s+(\d{1,2})\b",
        r"\bребята\s+(\d{1,2})\b",

        r"\bвсего\s+(?:будет\s+)?(?:нас\s+)?(\d{1,2})\b",
        r"\bвсего\s+(?:ребят|человек|игроков|участников)\s+(\d{1,2})\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            number = int(match.group(1))

            if valid_people_number(number) and not is_age_context(
                text,
                match.start(1),
                match.end(1)
            ):
                return number

    # 6. Неформальные кейсы
    informal_patterns = {
        r"\bя\s+и\s+(девушка|парень|жена|муж|друг|подруга|сын|дочь|ребенок|ребёнок)\b": 2,
        r"\bмы\s+с\s+(девушкой|парнем|женой|мужем|другом|подругой|сыном|дочкой|ребенком|ребёнком)\b": 2,

        r"\bпара\b": 2,
        r"\bвдвоем\b": 2,
        r"\bвдвоём\b": 2,

        r"\bвтроем\b": 3,
        r"\bвтроём\b": 3,

        r"\bвчетвером\b": 4,
        r"\bвпятером\b": 5,
        r"\bвшестером\b": 6,
        r"\bвсемером\b": 7,
        r"\bввосьмером\b": 8,
        r"\bвдевятером\b": 9,
        r"\bвдесятером\b": 10,
    }

    for pattern, number in informal_patterns.items():
        if re.search(pattern, text):
            return number

    return UNKNOWN