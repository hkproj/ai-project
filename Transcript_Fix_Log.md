# Log of manual transcript fix

## Transcript regularization

### 2023-04-08 - Regularization

```bash
2023-04-28 15:34:09,426 - data_processing - INFO - Video FC1_j3zJt20        - Avg: 23.514 c/s; Std: 15.369 c/s; Original words: 00561; Total removed: 00184 (032%)
2023-04-28 15:34:09,451 - data_processing - INFO - Video X7X3faMdjdI        - Avg: 17.048 c/s; Std: 14.239 c/s; Original words: 01150; Total removed: 00213 (018%)
2023-04-28 15:34:09,494 - data_processing - INFO - Video iqBOfND7BZY        - Avg: 19.666 c/s; Std: 16.085 c/s; Original words: 01411; Total removed: 00135 (009%)
2023-04-28 15:34:09,548 - data_processing - INFO - Video 1ULE81gYxmA        - Avg: 21.261 c/s; Std: 14.608 c/s; Original words: 00813; Total removed: 00138 (016%)
2023-04-28 15:34:09,632 - data_processing - INFO - Video DbmeHqQuoeY        - Avg: 20.263 c/s; Std: 13.436 c/s; Original words: 03346; Total removed: 00506 (015%)
2023-04-28 15:34:09,998 - data_processing - INFO - Video trPGQle76so        - Avg: 19.376 c/s; Std: 13.626 c/s; Original words: 01682; Total removed: 00229 (013%)
2023-04-28 15:34:10,065 - data_processing - INFO - Video 88usrrJ_UjY        - Avg: 24.341 c/s; Std: 13.262 c/s; Original words: 02105; Total removed: 00687 (032%)
2023-04-28 15:34:10,078 - data_processing - INFO - Video fixed-4XVg57NjwM   - Avg: 15.539 c/s; Std: 11.105 c/s; Original words: 00742; Total removed: 00035 (004%)
2023-04-28 15:34:10,109 - data_processing - INFO - Video C58taMCcnFA        - Avg: 21.138 c/s; Std: 14.494 c/s; Original words: 01829; Total removed: 00408 (022%)
2023-04-28 15:34:10,204 - data_processing - INFO - Video HfyGwiepb1s        - Avg: 22.263 c/s; Std: 13.701 c/s; Original words: 02396; Total removed: 00540 (022%)
2023-04-28 15:34:10,463 - data_processing - INFO - Video DbvpUxTkUH4        - Avg: 21.237 c/s; Std: 15.400 c/s; Original words: 01981; Total removed: 00533 (026%)
2023-04-28 15:34:11,041 - data_processing - INFO - Video OznYAHlGFwQ        - Avg: 21.786 c/s; Std: 17.395 c/s; Original words: 06219; Total removed: 01844 (029%)
2023-04-28 15:34:11,059 - data_processing - INFO - Video 1Wf9e4-v3aE        - Avg: 11.949 c/s; Std: 15.649 c/s; Original words: 01152; Total removed: 00113 (009%)
2023-04-28 15:34:11,077 - data_processing - INFO - Video H5u-zv0y8J4        - Avg: 16.931 c/s; Std: 13.177 c/s; Original words: 00682; Total removed: 00041 (006%)
2023-04-28 15:34:11,109 - data_processing - INFO - Video 0Hsq909zQYE        - Avg: 20.206 c/s; Std: 15.278 c/s; Original words: 01455; Total removed: 00231 (015%)
2023-04-28 15:34:11,196 - data_processing - INFO - Video HpAZRXr3Avw        - Avg: 19.149 c/s; Std: 13.642 c/s; Original words: 00806; Total removed: 00110 (013%)
2023-04-28 15:34:11,319 - data_processing - INFO - Video 2rAFcXYPmeQ        - Avg: 21.035 c/s; Std: 16.088 c/s; Original words: 01508; Total removed: 00344 (022%)
2023-04-28 16:43:27,948 - data_processing - INFO - Video Jb_VMX-XrI8        - Avg: 22.761 c/s; Std: 16.927 c/s; Original words: 00883; Total removed: 00341 (038%)
```

## Transcript cleaning

### 2023-04-28 - Cleaning

```bash
2023-04-28 15:36:50,157 - data_processing - WARNING - Video iqBOfND7BZY - Clip 0_4 - Found invalid char at index 7: caz*****e

caz*****e --> cazzate

2023-04-28 15:36:50,384 - data_processing - WARNING - Video HfyGwiepb1s - Clip 0_25 - Found floating point number at index 33: ADC 3.0.

ADC 3.0. --> adc tre punto zero

2023-04-28 15:36:50,449 - data_processing - WARNING - Video OznYAHlGFwQ - Clip 0_48 - Found invalid char at index 21: q&a

q&a --> q and a

2023-04-28 15:36:50,458 - data_processing - WARNING - Video OznYAHlGFwQ - Clip 0_65 - Found invalid char at index 229: q&a

q&a --> q and a

2023-04-28 15:36:50,479 - data_processing - WARNING - Video OznYAHlGFwQ - Clip 0_37 - Found invalid char at index 114: q&a

q&a --> q and a

2023-04-28 15:36:50,549 - data_processing - WARNING - Video H5u-zv0y8J4 - Clip 0_2 - Found invalid char at index 29: al cento%

The original text was 'al 100%' --> al cento percento 

```
