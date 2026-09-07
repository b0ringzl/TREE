import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        treeCoordinateMode: resolve(process.cwd(), "index.html"),
        frameSequenceMode: resolve(process.cwd(), "frame.html"),
      },
    },
  },
});
