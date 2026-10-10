# CrisisMind — Manual Testing Package

This folder contains everything needed to manually verify every CrisisMind feature before presentation.

## Contents

| Path | Description |
|---|---|
| `sample_batch.csv` | 20-row CSV for **Batch Processing** (balanced: 7 little, 6 mild, 7 severe) |
| `sample_images/` | 21 images from CrisisMMD v2.0 for batch / single analysis |
| `manual_test/` | 15 images (5 little, 5 mild, 5 severe) for quick manual verification |
| `manual_text_samples.md` | 15 text descriptions (5 per damage class) |
| `TEST_CHECKLIST.md` | Feature-by-feature testing guide with pass/fail boxes |
| `API_TESTS.md` | Every API endpoint with method, URL, input, expected response |
| `KNOWN_LIMITATIONS.md` | Known limitations and caveats |

## Image Class Mapping

### `sample_images/` — Little Damage (7)

| File | Disaster Event |
|---|---|
| `906984251538776071_0.jpg` | Hurricane Harvey |
| `907296821357289476_0.jpg` | Hurricane Harvey |
| `907648877460955137_0.jpg` | Hurricane Harvey |
| `909813776236650496_0.jpg` | Hurricane Irma |
| `909879652306575360_0.jpg` | Hurricane Irma |
| `910627521732972544_0.jpg` | Hurricane Maria |
| `916310521661591552_0.jpg` | Hurricane Maria |

### `sample_images/` — Mild Damage (7)

| File | Disaster Event |
|---|---|
| `907252873565757441_1.jpg` | Hurricane Harvey |
| `909757909944369152_0.jpg` | Hurricane Irma |
| `910614931262681089_0.jpg` | Mexico Earthquake |
| `912288130354511873_0.jpg` | Hurricane Maria |
| `914555402737668097_0.jpg` | Hurricane Maria |
| `917933078969966592_0.jpg` | California Wildfires |
| `921421609726029825_0.jpg` | Hurricane Maria |

### `sample_images/` — Severe Damage (7)

| File | Disaster Event |
|---|---|
| `905153821369917440_0.jpg` | Hurricane Harvey |
| `905636901306163200_0.jpg` | Hurricane Irma |
| `906023333044424708_0.jpg` | Hurricane Harvey |
| `909945268145197059_0.jpg` | Hurricane Harvey |
| `917975647116513280_0.jpg` | California Wildfires |
| `921428147861708800_0.jpg` | Hurricane Maria |
| `931304735029846017_0.jpg` | Iraq Iran Earthquake |

### `manual_test/` — Little Damage (5)

| File | Disaster Event |
|---|---|
| `901835409440768000_1.jpg` | Hurricane Harvey |
| `905230326359891968_0.jpg` | Hurricane Harvey |
| `909742125662789632_0.jpg` | Hurricane Irma |
| `910110509852184576_0.jpg` | Hurricane Irma |
| `910691589034409984_0.jpg` | Mexico Earthquake |

### `manual_test/` — Mild Damage (5)

| File | Disaster Event |
|---|---|
| `905686913876582401_0.jpg` | Hurricane Irma |
| `906664753602457601_0.jpg` | Hurricane Harvey |
| `907638041661108224_0.jpg` | Hurricane Harvey |
| `909740728909205504_0.jpg` | Hurricane Irma |
| `920016499473047552_0.jpg` | California Wildfires |

### `manual_test/` — Severe Damage (5)

| File | Disaster Event |
|---|---|
| `870283105151668224_1.jpg` | Sri Lanka Floods |
| `901805287958601728_0.jpg` | Hurricane Harvey |
| `910162829461860352_0.jpg` | Hurricane Harvey |
| `910559723770040326_1.jpg` | Hurricane Maria |
| `920131252602142720_0.jpg` | California Wildfires |
