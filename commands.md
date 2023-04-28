# Commands

## Faces processing

### Extract faces

```bash
python data_processing.py extract-faces --workers 10 --branch BRANCH_NAME
```

### Merge faces

```bash
python data_processing.py merge-faces --video-id VIDEOID --target TARGET --source SOURCE1 SOURCE2 SOURCE3
```

### Create intervals

```bash
python data_processing.py create-intervals --video-id VIDEOID --target TARGET --max-diff 200 --warn-limit 2000
```

## Clip processing

### Create clips

```bash
python data_processing.py create-clips --video-id VIDEOID --target TARGET --min-duration 5
```

### Extract audio

```bash
python data_processing.py extract-audio --video-id VIDEOID
```

### Transcribe audio

```bash
python data_processing.py transcribe-audio --video-id VIDEOID
```

### Transcription processing

#### Irregular transcripts

```bash
python data_processing.py irregular-transcript --window-size 20 --warn-speed 25 --fix
```

#### Transcription cleaning

```bash
python data_processing.py clean-transcript
```

### Extract mini clips

```bash
python data_processing.py create-miniclips --video-id VIDEOID
```