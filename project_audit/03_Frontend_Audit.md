# 03 — Frontend Audit

## Overview

The frontend is a React 19 + TypeScript + Vite + TailwindCSS v4 application with 3 pages, 12 components, and 4 UI primitives.

## Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| React | ^19.2.7 | UI framework |
| TypeScript | ~6.0.2 | Type safety |
| Vite | ^8.1.0 | Build tool |
| TailwindCSS | ^4.3.1 | Styling |
| Framer Motion | ^12.42.0 | Animations |
| React Router DOM | ^7.18.0 | Routing |
| Axios | ^1.18.1 | HTTP client |
| Lucide React | ^1.21.0 | Icons |

## Build Artifacts (from `npm run build`)

- `dist/index.html` — 0.94 KB (gzip: 0.50 KB)
- `dist/assets/index-*.css` — 36.18 KB (gzip: 7.07 KB)
- `dist/assets/index-*.js` — 484.87 KB (gzip: 155.46 KB)

## Architecture

```
App.tsx (BrowserRouter)
  └── Layout (Navbar + Footer + motion)
       ├── Home → 7 sections (Hero, Stats, Features, Tech Stack, Timeline, Highlights, Coming Soon)
       ├── Analysis → 4-step workflow (Upload, Text Input, Analyze Button, Prediction Result)
       └── About → Card grid layout
```

## Component Audit

### Pages

| Page | Lines | States Covered | Issues |
|---|---|---|---|
| Home/index.tsx | ~210 | Scroll-triggered entrance only | No loading skeleton for stats/tech; `border-b` inconsistent on last section |
| Analysis/index.tsx | ~160 | Loading, error (with retry), empty, result | Fragile step logic (nested ternaries); no abort on unmount; `catch (e: any)` |
| About/index.tsx | ~175 | Scroll-triggered entrance only | `techIcons` Record defined but never rendered; `key={i}` in 2 maps |

### Shared Components

| Component | Reusable? | States | Issues |
|---|---|---|---|
| Navbar.tsx | ✅ Yes | Active route highlight, mobile open/close | GitHub SVG inlined 3x (2 here + Footer) |
| Footer.tsx | ✅ Yes | Static | GitHub SVG 4th copy |
| Layout.tsx | ✅ Yes | Page transition animation | Clean |
| FeatureCard.tsx | ✅ Yes | Hover, scroll entrance | Shares identical animation pattern with ComingSoonCard |
| ComingSoonCard.tsx | ✅ Yes | Disabled/grayscale | Same animation as FeatureCard (duplicated code) |
| UploadCard.tsx | ✅ Yes | Upload (drag/hover/click), preview, cleared | **Memory leak**: `URL.createObjectURL` never revoked; no reset on prop change |
| PredictionCard.tsx | ✅ Yes | Result display | Dataset/confidence threshold hardcoded |
| LoadingSpinner.tsx | ✅ Yes | Sequential messages | Clean interval management |
| ErrorAlert.tsx | ✅ Yes | Error/warning, retry, dismiss | If `message=""` but `visible=true`, renders empty |
| AnimatedCounter.tsx | ✅ Yes | Scroll-triggered animation | Uses `setInterval` instead of `requestAnimationFrame` |
| Timeline.tsx | ✅ Yes | Scroll entrance | Hardcoded stages; not keyboard-navigable |
| SystemStatus.tsx | ✅ Yes | Checking/Connected/Offline | Polls every 15s; model/dataset/version hardcoded |

### UI Primitives

| Component | Issues |
|---|---|
| button.tsx | No `class-variance-authority` usage despite it being installed; clean otherwise |
| card.tsx | Clean |
| badge.tsx | No variants (unlike Button) |
| skeleton.tsx | **Zero usages** in the entire app — dead component |

## Data Flow

```
User Input (image File + text string)
  → Analysis page state (useState)
  → services/api.ts (Axios POST /predict, FormData)
  → FastAPI backend
  → PredictResponse
  → PredictionCard (display)
  → ErrorAlert (on failure)
```

## State Management

- All state is local `useState` in Analysis page
- No global state management (no Context, no Redux, no Zustand)
- API health polling uses `useEffect` + `setInterval` in SystemStatus

## API Integration

| Endpoint | Method | Used In | Frequency |
|---|---|---|---|
| `GET /health` | Axios | SystemStatus.tsx | Every 15s |
| `POST /predict` | Axios (FormData) | Analysis/index.tsx | On submit |

## Issues

### Critical

| # | Issue | Location | Impact |
|---|---|---|---|
| C1 | **Memory leak**: `URL.createObjectURL` never revoked | UploadCard.tsx:17 | Each upload leaks a blob URL until tab closes |
| C2 | **No request cancellation** — no AbortController | services/api.ts + Analysis/index.tsx | Orphaned requests on unmount |
| C3 | **`success` field never validated** — UI shows result even if `success: false` | Analysis/index.tsx:38 | Silent data integrity failure |

### Code Quality

| # | Issue | Location |
|---|---|---|
| Q1 | Dead code: `techIcons` Record defined but never rendered | About/index.tsx:54-61 |
| Q2 | Dead file: `css.d.ts` (redundant with `vite-env.d.ts`) | src/css.d.ts |
| Q3 | Dead component: `Skeleton` zero usages | ui/skeleton.tsx |
| Q4 | `techStack` array duplicated identically | Home/index.tsx:18-25, About/index.tsx:63-70 |
| Q5 | GitHub SVG inlined 3x (2 in Navbar, 1 in Footer) | Multiple files |
| Q6 | Fragile step logic (nested ternaries) | Analysis/index.tsx:29 |
| Q7 | `catch (e: any)` loses type safety | Analysis/index.tsx:40 |
| Q8 | No error boundary in App.tsx | App.tsx |

### Unused Dependencies (5 packages)

| Package | Reason |
|---|---|
| `@radix-ui/react-dialog` | Never imported |
| `@radix-ui/react-toast` | Never imported |
| `@radix-ui/react-slot` | Never imported |
| `class-variance-authority` | Button implements variants manually |
| `react-hook-form` | Never imported |

### UI/UX

| # | Issue |
|---|---|
| U1 | No 404 route — unknown paths show blank page |
| U2 | No cancel button during loading |
| U3 | No dark mode support |
| U4 | No focus-visible outlines in index.css |
| U5 | No loading skeletons on Home page stats/tech/About cards |

## Overall Frontend Score: **7.5 / 10**
