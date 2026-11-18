import { defineConfig } from "vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";

// Produce a real prerendered TanStack React build. scripts/publish_static.py copies
// dist/client to frontend-dist. static_host.py can serve the build and original API.
export default defineConfig({
  server: { port: 3000, host: "127.0.0.1" },
  plugins: [
    tsconfigPaths(),
    tailwindcss(),
    tanstackStart({
      pages: ["/", "/graph", "/transactions", "/risk", "/simulation", "/model", "/settings"].map(
        (path) => ({ path }),
      ),
      prerender: { enabled: true, crawlLinks: true, autoSubfolderIndex: true },
    }),
    react(),
  ],
});
