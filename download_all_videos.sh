#!/bin/bash
while read in; do yt-dlp --verbose --paths "./data/raw_videos" -f "best[ext=mp4]" -o "%(id)s.%(ext)s" "$in"; done < videos_list.txt