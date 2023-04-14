import re
from num2words import num2words

CHARS_TO_REMOVE = ['.',',','?','!',':',';','-']

def transformCase(word: str) -> str:
    return word.lower()

def removeUselessChars(word: str) -> tuple[bool, str]:
    # Remove all special characters from this string
    anyCharRemoved = False
    newWord = word[:]
    for char in CHARS_TO_REMOVE:
        if char in word:
            newWord = newWord.replace(char, '')
            anyCharRemoved = True
    return anyCharRemoved, newWord

def convertAllNumbersToWords(word: str) -> tuple[bool, str]:
    NUMBER_REGEXP = r"\d+" # Numbers
    newWord = word[:]
    reg = re.compile(NUMBER_REGEXP)
    matches = reg.finditer(newWord)
    anyNumbersFound = False
    for match in matches:
        anyNumbersFound = True
        num = match.group()
        start, end = match.start(), match.end()
        newWord = newWord[:start] + number_to_italian_word(int(num)) + newWord[end:]
    return [anyNumbersFound, newWord]

def detectFloatingPointNumbers(word: str) -> bool:
    FLOATING_POINT_NUMBER_REGEXP = r"(\d*\.\d+)|(\d*\,\d+)" # Numbers like 0,1 - 0.1 or .1
    reg = re.compile(FLOATING_POINT_NUMBER_REGEXP)
    return reg.match(word) is not None

def detectInvalidChars(word: str) -> bool:
    VALID_CHARS_REGEXP = r"^[a-z 'èéàìóòù]*$" # Only letters, spaces and apostrophes
    reg = re.compile(VALID_CHARS_REGEXP)
    return reg.match(word) is None

def number_to_italian_word(number) -> str:
    return num2words(number, lang='it')

if __name__ == '__main__':
    while True:
        num = input('Write a number: ')
        if num == "-1":
            break
        else:
            try:
                n = float(num)
                try:
                    n = int(num)
                except (ValueError):
                    n = float(num)
                print(number_to_italian_word(n))
            except:
                print('Invalid number')