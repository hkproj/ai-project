#!/bin/bash

# Read the first parameter and save it into a variable
# This is the path to the directory where the files are located
videos_path=$1

# Iterate recursively through all the files in the videos_path variable and if it's an mp4 file, print its duration to the output along with the path
find $videos_path -type f -name "*.mp4" -print0 | while IFS= read -r -d '' file; do
    # Get the duration of the video file
    duration=$(/usr/bin/ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    # Print the duration and the full path of the file to stdout
    echo "$file $duration"
done



