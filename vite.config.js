import { defineConfig } from "vite";

/** @type {import('vite').UserConfig} */

export default defineConfig({
    base: "https://wardvanassche.github.io/PRG08GAME/",

    build: {
        outDir: 'docs',
        emptyOutDir: true, // empty the build dir before new build
    }
});