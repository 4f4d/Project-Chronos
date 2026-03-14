// Bug 47 note: After `npm audit fix --force` upgraded Vite 5→7,
// @vitejs/plugin-react must also be upgraded from ^4.x to ^5.x.
// Run: npm install @vitejs/plugin-react@^5 --save-dev
// (see package.json — devDependencies will be updated separately)
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
    // Bug 48 fix: process.env.VITE_API_URL is UNDEFINED inside vite.config.js
    // because VITE_-prefixed vars are NOT placed in process.env — they are only
    // available in import.meta.env at browser runtime. loadEnv() reads the .env
    // file explicitly so the proxy target resolves correctly in two-Mac setups.
    const env = loadEnv(mode, process.cwd(), "");
    const apiTarget = env.VITE_API_URL || "http://localhost:8000";

    return {
        plugins: [react()],
        server: {
            host: "0.0.0.0",   // Expose to LAN so the second Mac can connect
            port: 5173,
            proxy: {
                "/api": {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ""),
                },
                "/ws": {
                    target: apiTarget,
                    ws: true,   // WebSocket proxy for real-time push
                    changeOrigin: true,
                },
            },
        },
    };
});

