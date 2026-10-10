# Frontend Package Audit

## Used Packages (11)

| Package | Version | Purpose | Imported In |
|---|---|---|---|
| `@tailwindcss/vite` | ^4.3.1 | TailwindCSS Vite plugin | vite.config.ts |
| `axios` | ^1.18.1 | HTTP client | services/api.ts |
| `clsx` | ^2.1.1 | Class name utility | lib/utils.ts |
| `framer-motion` | ^12.42.0 | Animations | All components |
| `lucide-react` | ^1.21.0 | Icons | All pages/components |
| `react` | ^19.2.7 | UI framework | Core |
| `react-dom` | ^19.2.7 | DOM rendering | Core |
| `react-router-dom` | ^7.18.0 | Routing | App.tsx |
| `tailwind-merge` | ^3.6.0 | Tailwind conflict resolution | lib/utils.ts |
| `tailwindcss` | ^4.3.1 | CSS framework | Styling |
| `@vitejs/plugin-react` | ^6.0.2 | Vite React plugin | vite.config.ts |

## Unused Packages (5) — Recommended for Removal

| Package | Version | Reason |
|---|---|---|
| `@radix-ui/react-dialog` | ^1.1.17 | Never imported; no modal/dialog implemented |
| `@radix-ui/react-slot` | ^1.3.0 | Never imported; no Radix Slot usage |
| `@radix-ui/react-toast` | ^1.2.17 | Never imported; ErrorAlert uses custom component |
| `class-variance-authority` | ^0.7.1 | Never imported; Button implements variants manually |
| `react-hook-form` | ^7.80.0 | Never imported; form state uses plain `useState` |

## Unused UI Component (1) — Consider Removing or Using

| Component | File | Reason |
|---|---|---|
| `Skeleton` | src/components/ui/skeleton.tsx | Defined but never imported anywhere in the app |

## Dev Dependencies — All Used (7)

| Package | Purpose |
|---|---|
| `@types/node` | Node type definitions |
| `@types/react` | React type definitions |
| `@types/react-dom` | ReactDOM type definitions |
| `oxlint` | Linter (npm run lint) |
| `typescript` | TypeScript compiler |
| `vite` | Build tool |

## Recommended Cleanup

1. Run `npm uninstall @radix-ui/react-dialog @radix-ui/react-slot @radix-ui/react-toast class-variance-authority react-hook-form`
2. Delete `src/components/ui/skeleton.tsx` (or use it for loading states)
3. Run `npm audit` to check for vulnerabilities
4. Run `npm prune` to remove extraneous packages

## Bundle Size Impact

| Category | Current (with unused) | After Cleanup | Savings |
|---|---|---|---|
| Production JS | 484.87 KB (gzip: 155.46 KB) | ~470 KB (est) | ~15 KB |
| Dependencies | 16 runtime + 7 dev | 11 runtime + 7 dev | 5 fewer |
| Install time | ~25s | ~20s (est) | ~5s |
