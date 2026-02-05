// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://gabrielfruet.github.io',
  base: '/portfolio',
  vite: {
    plugins: [tailwindcss()]
  },

  integrations: [mdx({
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex]
  }), sitemap()]
});