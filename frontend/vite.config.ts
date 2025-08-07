import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig, loadEnv, splitVendorChunkPlugin } from "vite"
import { visualizer } from 'rollup-plugin-visualizer'
import viteCompression from 'vite-plugin-compression'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const isProduction = mode === 'production'
  const cdnEnabled = env.VITE_CDN_ENABLED === 'true'
  
  return {
    plugins: [
      react({
        // Enable Fast Refresh
        fastRefresh: command === 'serve',
        // JSX runtime optimization
        jsxRuntime: 'automatic',
      }),
      
      // Smart vendor chunk splitting
      splitVendorChunkPlugin(),
      
      // Compression plugins for production
      isProduction && viteCompression({
        algorithm: 'gzip',
        ext: '.gz',
        threshold: 1024, // Only compress files > 1kb
        filter: /\.(js|css|html|svg|json)$/i,
        deleteOriginalAssets: false,
      }),
      
      isProduction && viteCompression({
        algorithm: 'brotliCompress',
        ext: '.br',
        threshold: 1024,
        filter: /\.(js|css|html|svg|json)$/i,
        deleteOriginalAssets: false,
      }),
      
      // Bundle analyzer in production
      isProduction && visualizer({
        filename: 'dist/bundle-analysis.html',
        open: false, // Don't auto-open in CI
        gzipSize: true,
        brotliSize: true,
        template: 'treemap', // Use treemap for better visualization
      }),
      
      // Additional optimization plugin for production
      isProduction && {
        name: 'chunk-split-plugin',
        generateBundle(options, bundle) {
          // Log chunk sizes for monitoring
          Object.keys(bundle).forEach(fileName => {
            const chunk = bundle[fileName];
            if (chunk.type === 'chunk' && chunk.code) {
              const size = new Blob([chunk.code]).size;
              console.log(`Chunk: ${fileName}, Size: ${(size / 1024).toFixed(2)}KB`);
            }
          });
        }
      }
    ].filter(Boolean),
    
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    
    build: {
      // Optimize for modern browsers
      target: ['es2020', 'edge88', 'firefox78', 'chrome87', 'safari12'],
      minify: 'terser',
      sourcemap: command === 'serve' || env.VITE_SOURCE_MAP === 'true',
      
      // CDN and asset optimization
      assetsDir: 'assets',
      reportCompressedSize: false, // Skip gzip reporting for faster builds
      
      // Code splitting configuration
      rollupOptions: {
        // Optimize imports
        external: cdnEnabled ? [
          // Externalize large libraries if using CDN
          ...(env.VITE_EXTERNAL_DEPS ? env.VITE_EXTERNAL_DEPS.split(',') : [])
        ] : [],
        
        output: {
          // Advanced chunk naming strategy
          chunkFileNames: (chunkInfo) => {
            // Extract meaningful names from modules
            const name = chunkInfo.name || 'chunk';
            
            // Handle vendor chunks
            if (name.includes('node_modules') || chunkInfo.isDynamicEntry === false) {
              return `assets/vendor/[name]-[hash].js`;
            }
            
            // Route-based chunks
            if (name.includes('pages') || name.includes('routes')) {
              return `assets/pages/[name]-[hash].js`;
            }
            
            // Component chunks
            if (name.includes('components')) {
              return `assets/components/[name]-[hash].js`;
            }
            
            return `assets/chunks/[name]-[hash].js`;
          },
          
          // Optimized asset naming
          assetFileNames: (assetInfo) => {
            if (!assetInfo.name) return 'assets/[name]-[hash].[ext]';
            
            const info = assetInfo.name.split('.');
            const ext = info[info.length - 1];
            
            // Categorize assets for better caching
            if (/\.(css)$/i.test(assetInfo.name)) {
              return `assets/css/[name]-[hash].css`;
            }
            if (/\.(png|jpe?g|svg|gif|webp|avif)$/i.test(assetInfo.name)) {
              return `assets/images/[name]-[hash].[ext]`;
            }
            if (/\.(woff2?|eot|ttf|otf)$/i.test(assetInfo.name)) {
              return `assets/fonts/[name]-[hash].[ext]`;
            }
            if (/\.(mp4|webm|ogg|mp3|wav|flac|aac)$/i.test(assetInfo.name)) {
              return `assets/media/[name]-[hash].[ext]`;
            }
            if (/\.(json|xml|txt)$/i.test(assetInfo.name)) {
              return `assets/data/[name]-[hash].[ext]`;
            }
            
            return `assets/misc/[name]-[hash].[ext]`;
          },
          
          // Advanced manual chunks for optimal caching
          manualChunks: (id) => {
            // Vendor chunk splitting by size and usage
            if (id.includes('node_modules')) {
              // Core React libraries - highest priority
              if (id.includes('react') || id.includes('react-dom')) {
                return 'react-core';
              }
              
              // Large UI libraries
              if (id.includes('@radix-ui') || id.includes('lucide-react')) {
                return 'ui-components';
              }
              
              // Animation and visualization
              if (id.includes('framer-motion') || id.includes('recharts') || id.includes('@react-spring')) {
                return 'visualization';
              }
              
              // Real-time collaboration
              if (id.includes('yjs') || id.includes('y-websocket') || id.includes('y-protocols') || id.includes('socket.io')) {
                return 'collaboration';
              }
              
              // Form and validation
              if (id.includes('react-hook-form') || id.includes('zod') || id.includes('@hookform')) {
                return 'forms';
              }
              
              // Utilities and small libraries
              if (id.includes('clsx') || id.includes('date-fns') || id.includes('uuid') || id.includes('tailwind')) {
                return 'utilities';
              }
              
              // Default vendor chunk for remaining dependencies
              return 'vendor';
            }
            
            // Application code splitting
            if (id.includes('src/components/')) {
              // Split large component groups
              if (id.includes('AutomataCanvas') || id.includes('SimulationEngine') || id.includes('Enhanced')) {
                return 'automata-core';
              }
              if (id.includes('AI') || id.includes('Tutor') || id.includes('Assistant')) {
                return 'ai-features';
              }
              if (id.includes('Collaborative') || id.includes('Room')) {
                return 'collaboration-ui';
              }
              if (id.includes('Complexity') || id.includes('Pumping') || id.includes('Parser')) {
                return 'theory-components';
              }
              if (id.includes('ui/')) {
                return 'ui-primitives';
              }
            }
            
            // Route-based splitting
            if (id.includes('src/router/') || id.includes('src/pages/')) {
              return 'router';
            }
            
            // Services and utilities
            if (id.includes('src/services/') || id.includes('src/utils/')) {
              return 'services';
            }
            
            // Contexts and hooks
            if (id.includes('src/contexts/') || id.includes('src/hooks/')) {
              return 'app-state';
            }
          }
        },
      },
      
      // Enhanced Terser configuration
      terserOptions: {
        compress: {
          drop_console: isProduction,
          drop_debugger: isProduction,
          pure_funcs: isProduction ? ['console.log', 'console.info', 'console.debug'] : [],
          passes: 3, // More optimization passes
          unsafe_arrows: true,
          unsafe_methods: true,
          unsafe_proto: true,
          hoist_funs: true,
          hoist_vars: true,
        },
        mangle: {
          safari10: true,
          properties: {
            regex: /^_/, // Mangle private properties
          },
        },
        format: {
          safari10: true,
          comments: false, // Remove all comments
        },
        toplevel: true,
      },
      
      // Optimized asset handling
      assetsInlineLimit: 2048, // 2KB limit - balance between requests and bundle size
      cssCodeSplit: true,
      cssMinify: true,
      
      // Performance monitoring
      chunkSizeWarningLimit: 800, // Stricter warning limit
      
      // Advanced module preload
      modulePreload: {
        polyfill: true,
        resolveDependencies: (filename, deps, { hostId, hostType }) => {
          // Custom module preload logic
          return deps.filter(dep => !dep.includes('legacy'));
        },
      },
    },
    
    // Enhanced CSS optimization
    css: {
      devSourcemap: command === 'serve',
      postcss: {
        plugins: [
          require('tailwindcss'),
          require('autoprefixer'),
          ...(isProduction ? [
            require('cssnano')({
              preset: ['default', {
                discardComments: { removeAll: true },
                normalizeWhitespace: true,
                minifyFontValues: true,
                minifySelectors: true,
              }]
            })
          ] : [])
        ],
      },
    },
    
    // Development server
    server: {
      port: 3000,
      host: true,
      cors: true,
      hmr: {
        overlay: true,
      },
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '/api'),
        },
      },
    },
    
    // Preview server
    preview: {
      port: 3000,
      host: true,
      cors: true,
    },
    
    // Enhanced dependency optimization
    optimizeDeps: {
      // Include critical dependencies
      include: [
        'react',
        'react-dom',
        'react/jsx-runtime',
        'react-dom/client',
        
        // Core UI dependencies
        '@radix-ui/react-slot',
        'clsx',
        'tailwind-merge',
        
        // Frequently used utilities
        'date-fns',
        'uuid',
      ],
      
      // Exclude large or problematic dependencies
      exclude: [
        'framer-motion', // Large animation library - lazy load
        'recharts', // Chart library - lazy load
        'yjs', // Collaboration - lazy load
        '@react-spring/web', // Animation - lazy load
      ],
      
      // Force optimization for specific packages
      force: [
        'react',
        'react-dom',
      ],
      
      // ESBuild options for dependency optimization
      esbuildOptions: {
        target: 'es2020',
        supported: {
          'top-level-await': false,
        },
      },
    },
    
    // Worker optimization
    worker: {
      format: 'es',
      plugins: [],
    },
    
    // Define global constants
    define: {
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
      __CDN_URL__: JSON.stringify(env.VITE_CDN_URL || ''),
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
      __PROD__: JSON.stringify(isProduction),
      __DEV__: JSON.stringify(!isProduction),
    },
    
    // Environment variables
    envPrefix: 'VITE_',
    
    // Experimental features
    experimental: {
      renderBuiltUrl: (filename, { hostType }) => {
        // Custom URL rendering for different host types
        if (hostType === 'js') {
          return env.VITE_CDN_URL ? `${env.VITE_CDN_URL}/${filename}` : `/${filename}`;
        }
        return `/${filename}`;
      },
    },
  }
})

