import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// App.js contains JSX, so .js files under src/ are compiled as JSX.
export default defineConfig({
  plugins: [react({ include: /\.(js|jsx)$/ })],
  esbuild: { loader: "jsx", include: /src\/.*\.jsx?$/, exclude: [] },
  optimizeDeps: { esbuildOptions: { loader: { ".js": "jsx" } } },
});
