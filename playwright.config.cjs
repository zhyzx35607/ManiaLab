const { defineConfig } = require('@playwright/test');
const path = require('path');
const python = process.env.MANIALAB_TEST_PYTHON || path.resolve('.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const port = process.env.MANIALAB_TEST_PORT || '8977';
module.exports = defineConfig({
  testDir: './tests/browser',
  outputDir: './tests/artifacts/browser',
  workers: 1,
  timeout: 30000,
  use: { baseURL: `http://127.0.0.1:${port}`, headless: true, viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure', screenshot: 'only-on-failure',
    ...(process.env.MANIALAB_TEST_BROWSER ? { channel: process.env.MANIALAB_TEST_BROWSER } : {}) },
  webServer: {
    command: `"${python}" tests/browser/fixture_server.py --port ${port}`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
    timeout: 30000,
  },
});
