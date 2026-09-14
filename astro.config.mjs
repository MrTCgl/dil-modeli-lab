// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwindcss from "@tailwindcss/vite";

// Tamamen statik build: sunucu yok, API yok. Bütün hesap tarayıcıda.
export default defineConfig({
  integrations: [react()],
  vite: { plugins: [tailwindcss()] },
});
