import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Builds straight into the Python package so `pip install` ships the GUI.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/local_llm_launcher/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
  },
})
