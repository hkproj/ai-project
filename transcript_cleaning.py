import re
from num2words import num2words

CHARS_TO_REMOVE = ['.',',','?','!',':',';','-']
REPLACEMENTS_FILE = 'transcript_replacements.txt'
VALID_CHARS_REGEXP = r"^[a-z 'èéàìíóòù]*$" # Only letters, spaces and apostrophes

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
    current = next(matches, None)
    while current is not None:
        anyNumbersFound = True
        num = current.group()
        start, end = current.start(), current.end()
        newWord = newWord[:start] + number_to_italian_word(int(num)) + newWord[end:]
        # Find again the matches (could be optimized, because we could just recalculate the positions of all the matches)
        matches = reg.finditer(newWord)
        current = next(matches, None)
    return [anyNumbersFound, newWord]

def performReplacementsFromFile(filePath: str, word: str) -> tuple[bool, str]:
    anyReplacements = False
    workingWord = word[:]
    with open(filePath, 'r') as f:
        lines = [line for line in f.readlines() if len(line.strip()) > 0]
        # The first line of each triplet of lines must be a comment
        for i in range(0, len(lines), 3):
            comment = lines[i].strip()
            if comment[0] != '#':
                raise Exception("The first line of each triplet of lines must be a comment")
            pattern = lines[i+1].strip('\n') # Preserve spaces!
            replacement = lines[i+2].strip('\n') # Preserve spaces!
            wordAfterReplacement = re.sub(pattern, replacement, workingWord)
            if workingWord != wordAfterReplacement:
                anyReplacements = True
            workingWord = wordAfterReplacement
    return (anyReplacements, workingWord)

def detectFloatingPointNumbers(word: str) -> bool:
    FLOATING_POINT_NUMBER_REGEXP = r"(\d*\.\d+)|(\d*\,\d+)" # Numbers like 0,1 - 0.1 or .1
    reg = re.compile(FLOATING_POINT_NUMBER_REGEXP)
    return reg.search(word) is not None

def detectInvalidChars(word: str) -> bool:
    reg = re.compile(VALID_CHARS_REGEXP)
    return reg.match(word) is None

def number_to_italian_word(number) -> str:
    return num2words(number, lang='it')