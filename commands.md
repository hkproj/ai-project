# Extract faces

```
python data_processing.py extract-faces --workers 10
```

# Merge faces

```
python data_processing.py merge-faces --video-id VIDEOID --target TARGET --source SOURCE
```

# Create intervals

```
python data_processing.py create-intervals --video-id VIDEOID --target TARGET --max-diff 200 --warn-limit 2000
```

# Create clips

```
python data_processing.py create-clips --video-id VIDEOID --target TARGET --min-duration 5
```

# Extract audio

```
python data_processing.py extract-audio --video-id VIDEOID
```