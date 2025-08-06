import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig, loadEnv } from "vite"
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const isProduction = mode === 'production'
  const cdnEnabled = env.VITE_CDN_ENABLED === 'true'
  
  return {
    plugins: [
      react(),
      // Bundle analyzer in production
      isProduction && visualizer({
        filename: 'dist/bundle-analysis.html',
        open: true,
        gzipSize: true,
        brotliSize: true,
      })
    ].filter(Boolean),
    
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    
    build: {
      // Optimize for production
      target: 'es2020',
      minify: 'terser',
      sourcemap: !isProduction,
      
      // CDN optimization
      assetsDir: 'assets',
      
      // Code splitting configuration
      rollupOptions: {
        output: {
          // Chunk naming strategy
          chunkFileNames: (chunkInfo) => {
            const facadeModuleId = chunkInfo.facadeModuleId ? chunkInfo.facadeModuleId.split('/').pop().replace('.tsx', '').replace('.ts', '') : 'chunk'
            return `assets/${facadeModuleId}-[hash].js`
          },
          assetFileNames: (assetInfo) => {
            const info = assetInfo.name.split('.')
            const ext = info[info.length - 1]
            
            // Group assets by type
            if (/\.(css)$/.test(assetInfo.name)) {
              return `assets/css/[name]-[hash].css`
            }
            if (/\.(png|jpe?g|svg|gif|tiff|bmp|ico)$/.test(assetInfo.name)) {
              return `assets/images/[name]-[hash].[ext]`
            }
            if (/\.(woff|woff2|eot|ttf|otf)$/.test(assetInfo.name)) {
              return `assets/fonts/[name]-[hash].[ext]`
            }
            return `assets/[name]-[hash].[ext]`
          },
          
          // Manual chunks for better caching
          manualChunks: {
            // Vendor chunks
            vendor: ['react', 'react-dom'],
            ui: ['@radix-ui/react-accordion', '@radix-ui/react-alert-dialog', '@radix-ui/react-avatar', '@radix-ui/react-button', '@radix-ui/react-card', '@radix-ui/react-checkbox', '@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-form', '@radix-ui/react-input', '@radix-ui/react-label', '@radix-ui/react-popover', '@radix-ui/react-select', '@radix-ui/react-separator', '@radix-ui/react-sheet', '@radix-ui/react-slider', '@radix-ui/react-switch', '@radix-ui/react-tabs', '@radix-ui/react-textarea', '@radix-ui/react-toast', '@radix-ui/react-tooltip'],
            chart: ['recharts'],
            animation: ['framer-motion'],
            collaboration: ['yjs', 'y-websocket', 'y-protocols'],
            websocket: ['socket.io-client'],
            utils: ['clsx', 'class-variance-authority', 'tailwind-merge', 'date-fns', 'zod', 'uuid']
          }
        },
        
        // External dependencies for CDN (optional)
        external: cdnEnabled ? [
          // Externalize large libraries if using CDN links
          // 'react',
          // 'react-dom'
        ] : []
      },
      
      // Terser options for better compression
      terserOptions: {
        compress: {
          drop_console: isProduction,
          drop_debugger: isProduction,
          passes: 2,
        },
        mangle: {
          safari10: true,
        },
        format: {
          safari10: true,
        },
      },
      
      // Asset optimization
      assetsInlineLimit: 4096, // 4kb limit for inlining
      cssCodeSplit: true,
      
      // Chunk size warnings
      chunkSizeWarningLimit: 1000,
    },
    
    // CSS optimization
    css: {
      devSourcemap: !isProduction,
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`,
        },
      },
    },
    
    // Development server
    server: {
      port: 3000,
      host: true,
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
    
    // Preview server (for production builds)
    preview: {
      port: 3000,
      host: true,
    },
    
    // Dependency optimization
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react/jsx-runtime',
        '@radix-ui/react-accordion',
        '@radix-ui/react-alert-dialog',
        '@radix-ui/react-avatar',
        'framer-motion',
        'recharts',
        'socket.io-client',
        'yjs',
      ],
      exclude: [
        // Exclude large dependencies that should be lazy loaded
      ],
    },
    
    // Define global constants
    define: {
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
      __CDN_URL__: JSON.stringify(env.VITE_CDN_URL || ''),
    },
  }
})

