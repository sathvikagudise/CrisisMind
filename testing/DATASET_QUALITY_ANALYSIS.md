# CrisisMind — Dataset Quality Analysis

## 1. Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Little or No Damage | 475 | 13.5% |
| Mild Damage | 839 | 23.8% |
| Severe Damage | 2212 | 62.7% |
| **Total** | **3526** | **100%** |

**Imbalance ratio:** 2212/475 = 4.7x

**Majority class baseline accuracy:** 62.7%

## 2. Duplicate Analysis

- Exact duplicates (text+image): 0


### Same Image, Different Labels

- Images with multiple labels: 0


### Same Text, Different Labels

- Texts with multiple labels: 104

  - [2x] labels={1: 1, 0: 1} text="#HurricaneHarvey almost up to my door..."
  - [4x] labels={2: 3, 1: 1} text="#PuertoRico has been obliterated by #HurricaneMaria it looks like they experienc..."
  - [3x] labels={1: 2, 2: 1} text="#Veyangalla, a village heavily affected by recent #floodSL in #Kalutura district..."
  - [3x] labels={2: 2, 1: 1} text=".@clarksvillepd Chaplain answering call for help in Puerto Rico -How you can too..."
  - [2x] labels={2: 1, 0: 1} text="6 more ways Hurricane Irma will cost Floridians via @usatoday..."
  - [2x] labels={1: 1, 2: 1} text="@CNN Help St. Croix United States Virgin Islands. Hurricane Maria hit them too!..."
  - [4x] labels={2: 3, 0: 1} text="@Reuters Poor beautiful #PuertoRico! So much damage. ὢD#HurricaneMaria #Requests..."
  - [4x] labels={1: 2, 0: 1, 2: 1} text="@Wolfdoggg MORE IMAGES CAME IN. St. Thomas and St. John. Please share..."
  - [2x] labels={1: 1, 2: 1} text="@realDonaldTrump Mr. President please stay focus and get your priorities straigh..."
  - [2x] labels={0: 1, 2: 1} text="A friend's front's front yard in #Austin #Texas this morning after high winds fr..."

## 3. Text-Image Cross-Annotation Agreement

- Keyword-label agreement: 380/671 (56.6%)

### Example Disagreements (label=Severe but text sounds mild/little)

- Label: Severe Damage — Text: "at least 11 dead and 100 missing as wildfires rage across northern california - damage and death tol..."
- Label: Severe Damage — Text: "this viral before-and-after photo shows the harrowing damage of california wildfires..."
- Label: Severe Damage — Text: "smoke, wildfires damage california's famed wine country..."
- Label: Severe Damage — Text: "damaged cell towers create communication problems in northern california fire zone..."
- Label: Severe Damage — Text: "haunting drone footage shows damage from tubbs fire in northern california..."

## 4. Text Characteristics

| Metric | Little | Mild | Severe |
|--------|--------|------|--------|
| Avg text length | 77 | 77 | 76 |
| Avg word count | 12 | 12 | 11 |
| % with URL | 0.0% | 0.0% | 0.0% |
| % with hashtag | 38.3% | 39.8% | 37.3% |
| % with mention | 27.6% | 28.6% | 25.6% |

## 5. Train/Val/Test Distribution

| Class | Train | Val | Test | Train% | Val% | Test% |
|-------|-------|-----|------|--------|------|-------|
| Little or No Damage | 333 | 71 | 71 | 13.5% | 13.4% | 13.4% |
| Mild Damage | 587 | 126 | 126 | 23.8% | 23.8% | 23.8% |
| Severe Damage | 1548 | 332 | 332 | 62.7% | 62.8% | 62.8% |

**Stratified split:** YES — proportions are consistent across splits.

## 6. Image File Analysis

- Missing image files: 0/3526

- Filename convention (_0=little, _1=severe) match: 500
- Filename convention mismatch: 2776


## 7. Little vs Mild Separability

### Text-based Separability

- Little ∩ Mild exact text overlap: 35
- Little ∩ Severe exact text overlap: 25
- Mild ∩ Severe exact text overlap: 58

**35 texts are labeled as both Little and Mild** — these are inherently ambiguous annotations.

### Image-based Separability

- Same image in Little and Mild: 0
- Same image in Little and Severe: 0
- Same image in Mild and Severe: 0

## 8. Per-Event Distribution

- Unique disaster events: 105


## 9. Annotation Quality Summary

### Issues Found

1. **104 texts with conflicting labels** — ambiguous annotations
2. **35 exact text overlaps between Little and Mild** — these classes may not be cleanly separable by text

### Impact on Model Performance

The dataset has **significant quality limitations**:

1. **Severe class dominance** (62.7%) creates inherent bias
2. **0 images with conflicting labels** introduce noise
3. **Little vs Mild overlap** (0 shared images, 35 shared texts) makes these classes inherently hard to separate
4. **Text style mismatch** — training uses tweets, inference uses clean English

**Conclusion:** A Macro F1 above ~0.58-0.60 may be unachievable without dataset improvements. The current 0.53-0.54 range is reasonable given the annotation quality.