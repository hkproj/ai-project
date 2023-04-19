import re

class SRTLoader:

    def __init__(self, filePath: str) -> None:
        self._parseFile(filePath)
    
    def _parseFile(self, filePath: str) -> None:
        with open(filePath, 'r') as file:
            lines = [line for line in file.readlines() if len(line.strip()) > 0]
            # Verify that the number of lines is divisible by 3
            if len(lines) % 3 != 0:           
                raise ValueError('Invalid SRT file (number of lines is not divisible by 3)')
            # Verify that the first line of each triplet is a number
            for i in range(0, len(lines), 3):
                if not lines[i].strip().isdigit():
                    raise ValueError(f'Invalid SRT file (first line of each triplet is not a number): {lines[i]}')
            # Verify that the second line of each triplet is in the format hh:mm:ss,mmm --> hh:mm:ss,mmm
            for i in range(1, len(lines), 3):
                if not re.match(r'\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d', lines[i]):
                    raise ValueError('Invalid SRT file (second line of each triplet is not in the format hh:mm:ss,mmm --> hh:mm:ss,mmm)')
            # Verify that the third line of each triplet is not empty
            for i in range(2, len(lines), 3):
                if len(lines[i].strip()) == 0:
                    raise ValueError('Invalid SRT file (third line of each triplet is empty)')
            # Parse the file
            self.entries = []
            for i in range(0, len(lines), 3):
                index = int(lines[i])
                if index != len(self.entries) + 1:
                    raise ValueError('Invalid SRT file (index is not sequential)')
                timestampString = lines[i+1]
                m = re.match(r'(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)', timestampString)
                if not m:
                    raise ValueError('Invalid SRT file (cannot extract timestamp from second line)')
                startH, startM, startS, startMS = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                endH, endM, endS, endMS = int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))
                startTimestamp = startH * 3600 + startM * 60 + startS + startMS / 1000
                endTimestamp = endH * 3600 + endM * 60 + endS + endMS / 1000
                words = lines[i+2].strip()

                self.entries.append((timestampString, startTimestamp, endTimestamp, words))
            

    def getAllWords(self) -> list[tuple[float, float, str]]:
        return [(originalTimestampString, startTimestamp, endTimestamp, words) for originalTimestampString, startTimestamp, endTimestamp, words in self.entries]
    
    def saveToFile(self, filePath: str, entries: list[tuple[float, float, str]]) -> None:
        with open(filePath, 'w') as file:
            for i, (originalTimestampString, startTimestamp, endTimestamp, words) in enumerate(entries):
                file.write(f'{i+1}\n')
                file.write(f'{originalTimestampString}')
                file.write(f'{words}\n')
                file.write('\n')