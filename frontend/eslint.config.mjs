import { FlatCompat } from "@eslint/eslintrc";

// eslint-config-next still ships legacy-style configs; FlatCompat adapts them.
const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const eslintConfig = [
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default eslintConfig;
