import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/upload': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        timeout: 600000, // 10 minutes for large videos
      },
      '/signup': 'http://127.0.0.1:8000',
      '/token': 'http://127.0.0.1:8000',
      '/history': 'http://127.0.0.1:8000',
      '/status': 'http://127.0.0.1:8000',
      '/geojson': 'http://127.0.0.1:8000',
      '/download': 'http://127.0.0.1:8000',
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
