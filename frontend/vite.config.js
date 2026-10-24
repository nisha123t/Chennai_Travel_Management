import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API calls go to /api/... and Vite forwards them to the FastAPI server on port 8000.
// This works the same on a laptop and in GitHub Codespaces (only port 5173 needs to be open).
const backend = {
  "/api": {
    target: "http://localhost:8000",
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true, proxy: backend },
  preview: { port: 4173, host: true, proxy: backend },
});
