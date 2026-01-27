import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['mcp-server/src/__tests__/**/*.test.ts'],
    environment: 'node',
  },
});
