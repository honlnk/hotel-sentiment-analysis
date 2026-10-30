import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// 开发时：npm run dev，前端跑在 Vite 服务上，/api 转发给 FastAPI
// 发布时：npm run build，产物输出到 ../static，由 FastAPI 统一托管
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
});
