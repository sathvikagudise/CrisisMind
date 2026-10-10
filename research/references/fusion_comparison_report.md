# Multimodal Fusion — Model Comparison Report

## Comparison Table

| Model         |   Accuracy |   Precision (macro) |   Recall (macro) |   F1 (macro) |   F1 (weighted) |
|:--------------|-----------:|--------------------:|-----------------:|-------------:|----------------:|
| ConcatFusion  |     0.6371 |              0.5332 |           0.5287 |       0.5302 |          0.64   |
| GatedFusion   |     0.6352 |              0.5329 |           0.5437 |       0.5375 |          0.6415 |
| ImageBaseline |     0.5841 |              0.4809 |           0.5063 |       0.4869 |          0.5959 |
| TextBaseline  |     0.525  |              0.4291 |           0.4817 |       0.4221 |          0.5328 |

**Best Model:** GatedFusion (by Macro F1)

## Per-Class Performance

### ConcatFusion

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| little_or_no_damage | 0.4308 | 0.3944 | 0.4118 | 71 |
| mild_damage | 0.3786 | 0.4206 | 0.3985 | 126 |
| severe_damage | 0.7901 | 0.7711 | 0.7805 | 332 |

### GatedFusion

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| little_or_no_damage | 0.3974 | 0.4366 | 0.4161 | 71 |
| mild_damage | 0.4058 | 0.4444 | 0.4242 | 126 |
| severe_damage | 0.7955 | 0.7500 | 0.7721 | 332 |

