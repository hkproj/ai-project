#!/bin/bash

while read in; do yt-dlp --verbose -o "%(id)s.%(ext)s" --format 18 "$in"; done < videos_list.csv