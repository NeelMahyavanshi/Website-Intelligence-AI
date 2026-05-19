import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ingest": "http://localhost:8001",
      "/query": "http://localhost:8001",
      "/companies": "http://localhost:8001",
      "/health": "http://localhost:8001",
    },
  },
});
