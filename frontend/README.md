# CrisisMind Frontend

Modern React dashboard for multimodal disaster severity classification.

## Tech Stack

- React 19 + TypeScript
- Vite
- TailwindCSS v4
- shadcn/ui primitives
- Framer Motion
- React Router DOM
- Axios
- Lucide React

## Setup

```bash
cd frontend
npm install
npm run dev
```

The dev server starts at `http://localhost:5173` and proxies `/predict` and `/health` to the FastAPI backend at `http://localhost:8000`.

## Backend

Ensure the FastAPI server is running:

```bash
cd ../backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Build

```bash
npm run build
```

Output goes to `dist/`.

## Project Structure

```
src/
├── components/
│   ├── ui/          # Button, Card, Badge, Skeleton
│   ├── Layout.tsx   # Page layout wrapper
│   ├── Navbar.tsx   # Sticky navigation
│   ├── Footer.tsx   # Site footer
│   ├── FeatureCard.tsx
│   ├── UploadCard.tsx
│   ├── PredictionCard.tsx
│   ├── ComingSoonCard.tsx
│   ├── LoadingSpinner.tsx
│   └── ErrorAlert.tsx
├── pages/
│   ├── Home/        # Landing page
│   ├── Analysis/    # Analysis dashboard
│   └── About/       # About page
├── services/
│   └── api.ts       # Axios API client
├── types/
│   └── index.ts     # TypeScript interfaces
├── lib/
│   └── utils.ts     # cn() helper
├── App.tsx
├── main.tsx
└── index.css
```
