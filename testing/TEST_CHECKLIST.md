# CrisisMind — Feature Test Checklist

Test each feature and mark **Pass** or **Fail**.

---

## 1. Home Page

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 1.1 | Page loads | Navigate to `/` | Hero section, features grid, tech stack, timeline all render | ☐ |
| 1.2 | "Start Analysis" link | Click button | Navigates to `/analysis` | ☐ |
| 1.3 | "Learn More" link | Click button | Scrolls to features section | ☐ |
| 1.4 | Completed Modules section | Scroll down | 4 green-badged cards visible (Explainable AI, Dashboard, Analysis Center, PDF Reports) | ☐ |
| 1.5 | Upcoming Modules section | Scroll further | 2 grey cards visible (Emergency Recommendation, Disaster Impact Index) | ☐ |
| 1.6 | Stats display | View hero | 4 stat counters shown | ☐ |

---

## 2. Manual Analysis — Image Only

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 2.1 | Upload image | Drag or click to upload an image from `manual_test/` (little_damage) | Image preview shown | ☐ |
| 2.2 | Select "Image Only" | Choose mode dropdown | Analysis mode set to Image Only | ☐ |
| 2.3 | Run analysis | Click "Analyze" button | Loading skeleton appears, results load within 15s | ☐ |
| 2.4 | Severity display | View results | Severity gauge shows a score (0–100) | ☐ |
| 2.5 | Confidence display | View results | Confidence percentage shown | ☐ |
| 2.6 | Disaster type | View results | Disaster type classification shown (e.g. "earthquake", "flood", "wildfire") | ☐ |
| 2.7 | Infrastructure impact | View results | Impact breakdown displayed | ☐ |
| 2.8 | Emergency level | View results | Emergency level badge shown | ☐ |
| 2.9 | Priority badge | View results | Priority (Low/Medium/High/Critical) shown | ☐ |
| 2.10 | Recommended actions | View results | Action list displayed | ☐ |
| 2.11 | Test mild damage image | Upload from `manual_test/` (mild) | Severity higher than little damage | ☐ |
| 2.12 | Test severe damage image | Upload from `manual_test/` (severe) | Severity higher than mild damage | ☐ |

---

## 3. Manual Analysis — Text Only

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 3.1 | Enter text | Paste sample from `manual_text_samples.md` (little damage) | Text displayed in input | ☐ |
| 3.2 | Select "Text Only" | Choose mode dropdown | Analysis mode set to Text Only | ☐ |
| 3.3 | Run analysis | Click "Analyze" button | Results load with text-based prediction | ☐ |
| 3.4 | Verify results | View severity/confidence | Assessment based on text description | ☐ |
| 3.5 | Test mild text | Use mild damage text sample | Severity higher than little | ☐ |
| 3.6 | Test severe text | Use severe damage text sample | Severity higher than mild | ☐ |

---

## 4. Manual Analysis — Multimodal

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 4.1 | Upload image + text | Upload image from `manual_test/` + paste matching text | Both inputs shown | ☐ |
| 4.2 | Select "Multimodal" | Choose mode | Mode set to Multimodal | ☐ |
| 4.3 | Run analysis | Click "Analyze" | Results use both modalities | ☐ |
| 4.4 | Compare with single-mode | Run same image in Image Only vs Multimodal | Multimodal may differ (fuses both inputs) | ☐ |

---

## 5. AI Explanation (Grad-CAM)

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 5.1 | Generate explanation | After analysis, click "AI Explanation" button | Loading, then 3 images appear (original, heatmap, overlay) | ☐ |
| 5.2 | Image-only explanation | First run Image Only, then request explanation | Heatmap highlights relevant regions | ☐ |
| 5.3 | Multimodal explanation | Run Multimodal, then request explanation | Heatmap generated from fused model | ☐ |
| 5.4 | Explanation download | Hover or check for download option | Images downloadable | ☐ |

---

## 6. Disaster Intelligence

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 6.1 | Navigate to Disasters | Click "Disasters" in nav | Page loads at `/disasters` | ☐ |
| 6.2 | View events | Existing events shown | Disaster feed displayed | ☐ |
| 6.3 | Create event | Click "New Event" | Form opens | ☐ |
| 6.4 | Analyze event | Fill form + submit | Analysis runs, results appear | ☐ |
| 6.5 | Severity badges | View event list | Color-coded badges shown | ☐ |

---

## 7. Dashboard

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 7.1 | Navigate to Dashboard | Click "Dashboard" in nav | Page loads at `/dashboard` | ☐ |
| 7.2 | Stat cards | View top section | 6 stat cards with numbers | ☐ |
| 7.3 | Severity distribution chart | Scroll to chart | Bar chart shown | ☐ |
| 7.4 | Disaster type breakdown | View chart | Pie/bar chart by disaster type | ☐ |
| 7.5 | Priority distribution | View chart | Distribution shown | ☐ |
| 7.6 | Confidence histogram | View chart | Histogram displayed | ☐ |
| 7.7 | Event timeline | View timeline | Chronological event list/chart | ☐ |
| 7.8 | Recent analyses | View table | Latest analyses listed | ☐ |
| 7.9 | System health | View section | Health metrics shown | ☐ |

---

## 8. Batch Processing

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 8.1 | Upload CSV | Navigate to `/batch`, upload `testing/sample_batch.csv` | CSV accepted, job ID returned | ☐ |
| 8.2 | Upload images | Upload ZIP of images from `testing/sample_images/` | Images accepted, job starts | ☐ |
| 8.3 | Monitor progress | Poll job status | Progress percentage increases | ☐ |
| 8.4 | View results | Wait for completion | Results table shown | ☐ |
| 8.5 | Verify batch output | Check severity columns | Varying severity across rows | ☐ |

---

## 9. Analysis Center

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 9.1 | Navigate | Click "Analysis Center" in nav | Page loads at `/analysis-center` | ☐ |
| 9.2 | View all records | Default view | Paginated table of all analyses | ☐ |
| 9.3 | Statistics cards | Top of page | 6 stat cards (total, by type, by severity, etc.) | ☐ |
| 9.4 | Detail modal | Click a row | Modal opens with full record details | ☐ |
| 9.5 | Sort control | Change sort dropdown | Table re-orders | ☐ |
| 9.6 | Pagination | Click next/prev page | Table pages correctly | ☐ |

---

## 10. Search

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 10.1 | Search bar | Type a query in Analysis Center | Results filtered | ☐ |
| 10.2 | Clear search | Clear search box | Full list restored | ☐ |
| 10.3 | Search by title | Type "earthquake" | Earthquake-related records shown | ☐ |
| 10.4 | Search by disaster type | Type "flood" | Flood records shown | ☐ |

---

## 11. Filters

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 11.1 | Filter by type | Select "manual", "batch", or "live" | Table filtered | ☐ |
| 11.2 | Filter by severity | Select severity tier (e.g. "severe") | Filtered results | ☐ |
| 11.3 | Filter by priority | Select priority level | Filtered results | ☐ |
| 11.4 | Combined filters | Apply type + severity + priority | All filters applied | ☐ |
| 11.5 | Clear filters | Reset filter panel | Full list restored | ☐ |

---

## 12. PDF Report

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 12.1 | Generate PDF | In Analysis Center, open detail modal, click "Download Report" | PDF downloads | ☐ |
| 12.2 | PDF content | Open downloaded PDF | Header, severity, confidence, actions, recommendation all present | ☐ |
| 12.3 | Grad-CAM in PDF | If explanation exists | Overlay image embedded in PDF | ☐ |

---

## 13. CSV Export

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 13.1 | Export all | Click "Export CSV" in Analysis Center | CSV file downloads | ☐ |
| 13.2 | Export filtered | Apply filter, then export | CSV contains only filtered rows | ☐ |
| 13.3 | CSV content | Open in spreadsheet | Columns: analysis_id, type, severity, confidence, disaster_type, source, timestamp | ☐ |

---

## 14. Navigation

| # | Test | How to Test | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 14.1 | All nav links | Click each nav item | Correct route loads | ☐ |
| 14.2 | Responsive | Resize browser | Mobile menu works | ☐ |

---

## Summary

| Total Tests | Pass | Fail |
|---|---|---|
| (fill after testing) | ☐ | ☐ |
