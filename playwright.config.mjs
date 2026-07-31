import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'tests/browser',use:{browserName:'chromium',viewport:{width:1440,height:1000}},reporter:'line'});
