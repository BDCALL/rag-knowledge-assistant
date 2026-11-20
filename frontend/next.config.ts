import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits a self-contained server bundle, which keeps a future Docker image small.
  output: "standalone",
};

export default nextConfig;
