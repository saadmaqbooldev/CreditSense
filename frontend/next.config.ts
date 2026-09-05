import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
    return [
      { source: "/health", destination: `${backendUrl}/health` },
      { source: "/predict/:path*", destination: `${backendUrl}/predict/:path*` },
      { source: "/submit/:path*", destination: `${backendUrl}/submit/:path*` },
    ];
  },
};

export default nextConfig;
