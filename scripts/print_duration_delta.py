import os
import sys

if __name__ == '__main__':
    # Read the first argument as the name of the first file
    file1 = os.path.abspath(sys.argv[1])
    # Read the second argument as the name of the second file
    file2 = os.path.abspath(sys.argv[2])

    # Read all the lines from file1
    with open(file1, 'r') as f:
        lines1 = [line for line in f.readlines() if len(line.split()) > 0]
    
    # Read all the lines from file2
    with open(file2, 'r') as f:
        lines2 = [line for line in f.readlines() if len(line.split()) > 0]

    # Make sure both files have the same number of lines
    assert len(lines1) == len(lines2)

    print(f'{"file":<50}\tduration_diff')

    # Iterate over the lines and print the difference
    for line1, line2 in zip(lines1, lines2):
        # Split the lines by spaces
        line1 = line1.split()
        line2 = line2.split()

        # Make sure both lines have the same number of elements
        assert len(line1) == len(line2)
        assert line1[0] == line2[0]

        # Get the duration of the first line
        duration1 = float(line1[1])
        # Get the duration of the second line
        duration2 = float(line2[1])

        # Print the difference
        print(f"{line1[0]:<50}\t{abs(duration2 - duration1):.3f}")
