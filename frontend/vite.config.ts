import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/predict": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
      "/disasters/events": { target: "http://localhost:8000", changeOrigin: true },
      "/batch/upload": { target: "http://localhost:8000", changeOrigin: true },
      "/batch/start": { target: "http://localhost:8000", changeOrigin: true },
      "/batch/status": { target: "http://localhost:8000", changeOrigin: true },
      "/batch/results": { target: "http://localhost:8000", changeOrigin: true },
      "/batch/download": { target: "http://localhost:8000", changeOrigin: true },
      "/dashboard/summary": { target: "http://localhost:8000", changeOrigin: true },
      "/dashboard/statistics": { target: "http://localhost:8000", changeOrigin: true },
      "/dashboard/charts": { target: "http://localhost:8000", changeOrigin: true },
      "/dashboard/timeline": { target: "http://localhost:8000", changeOrigin: true },
      "/dashboard/recent": { target: "http://localhost:8000", changeOrigin: true },
      "/report": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
