import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export served by FastAPI (backend/app/main.py). trailingSlash emits run/index.html,
  // which Starlette's StaticFiles resolves for /run/.
  output: "export",
  trailingSlash: true,
};

export default nextConfig;
