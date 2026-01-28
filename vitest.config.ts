import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['mcp-server/src/__tests__/**/*.test.ts'],
    environment: 'node',
    coverage: {
      provider: 'v8',
      include: ['mcp-server/src/**/*.ts'],
      exclude: ['mcp-server/src/__tests__/**'],
      reporter: ['text', 'text-summary'],
    },
  },
});
