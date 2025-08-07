/**
 * Advanced Webpack Configuration with Module Federation and Optimal Code Splitting
 * Alternative to Vite for enterprise-grade applications requiring micro-frontends
 */
const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const ModuleFederationPlugin = require('@module-federation/webpack');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');
const ESLintPlugin = require('eslint-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');
const WorkboxPlugin = require('workbox-webpack-plugin');

const isDevelopment = process.env.NODE_ENV !== 'production';
const isAnalyze = process.env.ANALYZE === 'true';

// Shared dependencies for module federation
const sharedDependencies = {
  react: {
    singleton: true,
    requiredVersion: '^18.3.1',
    eager: true
  },
  'react-dom': {
    singleton: true,
    requiredVersion: '^18.3.1',
    eager: true
  },
  'react-router-dom': {
    singleton: true,
    requiredVersion: '^6.0.0'
  },
  '@radix-ui/react-slot': {
    singleton: true,
    requiredVersion: '^1.2.3'
  },
  'tailwind-merge': {
    singleton: true,
    requiredVersion: '^3.0.0'
  },
  'clsx': {
    singleton: true,
    requiredVersion: '^2.0.0'
  }
};

// Advanced chunk splitting strategy
function createSplitChunksConfig() {
  return {
    chunks: 'all',
    minSize: 20000,
    maxSize: 200000,
    minChunks: 1,
    maxAsyncRequests: 30,
    maxInitialRequests: 30,
    cacheGroups: {
      // Critical vendor chunk - React ecosystem
      reactVendor: {
        test: /[\\/]node_modules[\\/](react|react-dom|react-router|react-router-dom)[\\/]/,
        name: 'vendor-react',
        chunks: 'all',
        priority: 40,
        enforce: true,
        reuseExistingChunk: true
      },
      
      // UI components - Radix UI
      uiVendor: {
        test: /[\\/]node_modules[\\/]@radix-ui[\\/]/,
        name: 'vendor-ui',
        chunks: 'all',
        priority: 35,
        reuseExistingChunk: true
      },
      
      // Animation and visualization libraries
      visualizationVendor: {
        test: /[\\/]node_modules[\\/](framer-motion|recharts|@react-spring)[\\/]/,
        name: 'vendor-visualization',
        chunks: 'async',
        priority: 30,
        reuseExistingChunk: true
      },
      
      // Real-time collaboration
      collaborationVendor: {
        test: /[\\/]node_modules[\\/](yjs|y-websocket|y-protocols|socket\.io-client)[\\/]/,
        name: 'vendor-collaboration',
        chunks: 'async',
        priority: 25,
        reuseExistingChunk: true
      },
      
      // Form handling
      formsVendor: {
        test: /[\\/]node_modules[\\/](react-hook-form|@hookform|zod)[\\/]/,
        name: 'vendor-forms',
        chunks: 'async',
        priority: 20,
        reuseExistingChunk: true
      },
      
      // Utility libraries
      utilsVendor: {
        test: /[\\/]node_modules[\\/](date-fns|uuid|lodash|ramda)[\\/]/,
        name: 'vendor-utils',
        chunks: 'all',
        priority: 15,
        reuseExistingChunk: true
      },
      
      // Remaining vendor libraries
      vendor: {
        test: /[\\/]node_modules[\\/]/,
        name: 'vendor',
        chunks: 'all',
        priority: 10,
        reuseExistingChunk: true
      },
      
      // Application code splitting by feature
      automataCore: {
        test: /[\\/]src[\\/]components[\\/](.*Automata.*|.*Canvas.*|.*Simulation.*)[\\/]/,
        name: 'app-automata-core',
        chunks: 'async',
        priority: 8,
        minChunks: 1,
        reuseExistingChunk: true
      },
      
      aiFeatures: {
        test: /[\\/]src[\\/]components[\\/](.*AI.*|.*Tutor.*|.*Assistant.*)[\\/]/,
        name: 'app-ai-features',
        chunks: 'async',
        priority: 7,
        minChunks: 1,
        reuseExistingChunk: true
      },
      
      collaboration: {
        test: /[\\/]src[\\/]components[\\/](.*Collaborative.*|.*Room.*|.*Workspace.*)[\\/]/,
        name: 'app-collaboration',
        chunks: 'async',
        priority: 6,
        minChunks: 1,
        reuseExistingChunk: true
      },
      
      theory: {
        test: /[\\/]src[\\/]components[\\/](.*Theory.*|.*Pumping.*|.*Parser.*|.*Complexity.*)[\\/]/,
        name: 'app-theory',
        chunks: 'async',
        priority: 5,
        minChunks: 1,
        reuseExistingChunk: true
      },
      
      // Common application chunks
      common: {
        test: /[\\/]src[\\/]/,
        name: 'app-common',
        chunks: 'all',
        priority: 1,
        minChunks: 2,
        reuseExistingChunk: true
      }
    }
  };
}

// Performance optimization rules
function createOptimizationConfig() {
  const config = {
    splitChunks: createSplitChunksConfig(),
    runtimeChunk: 'single',
    
    // Module concatenation for better performance
    concatenateModules: !isDevelopment,
    
    // Tree shaking
    usedExports: true,
    providedExports: true,
    sideEffects: false,
    
    // Module IDs for better caching
    moduleIds: isDevelopment ? 'named' : 'deterministic',
    chunkIds: isDevelopment ? 'named' : 'deterministic',
  };
  
  if (!isDevelopment) {
    config.minimizer = [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true,
            drop_debugger: true,
            pure_funcs: ['console.log', 'console.info', 'console.debug'],
            passes: 3,
            unsafe_arrows: true,
            unsafe_methods: true,
            unsafe_proto: true,
            hoist_funs: true,
            hoist_vars: true
          },
          mangle: {
            safari10: true,
            properties: {
              regex: /^_/
            }
          },
          format: {
            safari10: true,
            comments: false
          },
          toplevel: true
        },
        parallel: true,
        extractComments: false
      }),
      new CssMinimizerPlugin({
        minimizerOptions: {
          preset: [
            'default',
            {
              discardComments: { removeAll: true },
              normalizeWhitespace: true,
              minifyFontValues: true,
              minifySelectors: true
            }
          ]
        }
      })
    ];
  }
  
  return config;
}

// Advanced module resolution
function createResolveConfig() {
  return {
    extensions: ['.tsx', '.ts', '.jsx', '.js', '.json'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@components': path.resolve(__dirname, 'src/components'),
      '@utils': path.resolve(__dirname, 'src/utils'),
      '@services': path.resolve(__dirname, 'src/services'),
      '@contexts': path.resolve(__dirname, 'src/contexts'),
      '@hooks': path.resolve(__dirname, 'src/hooks'),
      '@types': path.resolve(__dirname, 'src/types')
    },
    fallback: {
      // Polyfills for Node.js modules in browser
      path: require.resolve('path-browserify'),
      os: require.resolve('os-browserify'),
      crypto: require.resolve('crypto-browserify'),
      stream: require.resolve('stream-browserify'),
      buffer: require.resolve('buffer')
    },
    // Optimize module resolution
    symlinks: false,
    cacheWithContext: false,
    // Prefer ES6 modules
    mainFields: ['browser', 'module', 'main']
  };
}

// Development server configuration
function createDevServerConfig() {
  return {
    port: 3000,
    host: '0.0.0.0',
    hot: true,
    liveReload: false, // Use HMR instead
    compress: true,
    historyApiFallback: {
      disableDotRule: true,
      rewrites: [
        { from: /.*/, to: '/index.html' }
      ]
    },
    proxy: [
      {
        context: ['/api'],
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      }
    ],
    static: {
      directory: path.join(__dirname, 'public'),
      publicPath: '/'
    },
    client: {
      overlay: {
        errors: true,
        warnings: false
      },
      progress: true
    }
  };
}

// Main webpack configuration
module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  
  return {
    mode: isDevelopment ? 'development' : 'production',
    
    entry: {
      main: './src/main.tsx'
    },
    
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction 
        ? 'assets/js/[name].[contenthash:8].js'
        : 'assets/js/[name].js',
      chunkFilename: isProduction
        ? 'assets/js/[name].[contenthash:8].chunk.js'
        : 'assets/js/[name].chunk.js',
      assetModuleFilename: 'assets/[hash][ext][query]',
      publicPath: '/',
      clean: true,
      // Enable cross-origin loading for module federation
      crossOriginLoading: 'anonymous'
    },
    
    resolve: createResolveConfig(),
    
    module: {
      rules: [
        // TypeScript and JSX
        {
          test: /\.(ts|tsx)$/,
          exclude: /node_modules/,
          use: [
            {
              loader: 'ts-loader',
              options: {
                transpileOnly: true, // Speed up builds
                configFile: 'tsconfig.json'
              }
            }
          ]
        },
        
        // CSS and PostCSS
        {
          test: /\.css$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            {
              loader: 'css-loader',
              options: {
                importLoaders: 1,
                sourceMap: isDevelopment
              }
            },
            {
              loader: 'postcss-loader',
              options: {
                postcssOptions: {
                  plugins: [
                    require('tailwindcss'),
                    require('autoprefixer'),
                    ...(isProduction ? [require('cssnano')] : [])
                  ]
                }
              }
            }
          ]
        },
        
        // Images and assets
        {
          test: /\.(png|jpe?g|gif|svg|webp|avif)$/i,
          type: 'asset',
          parser: {
            dataUrlCondition: {
              maxSize: 8 * 1024 // 8kb
            }
          },
          generator: {
            filename: 'assets/images/[name].[contenthash:8][ext]'
          }
        },
        
        // Fonts
        {
          test: /\.(woff|woff2|eot|ttf|otf)$/i,
          type: 'asset/resource',
          generator: {
            filename: 'assets/fonts/[name].[contenthash:8][ext]'
          }
        },
        
        // Other assets
        {
          test: /\.(mp4|webm|ogg|mp3|wav|flac|aac)$/i,
          type: 'asset/resource',
          generator: {
            filename: 'assets/media/[name].[contenthash:8][ext]'
          }
        }
      ]
    },
    
    plugins: [
      // Module Federation for micro-frontends
      new ModuleFederationPlugin({
        name: 'automataApp',
        filename: 'remoteEntry.js',
        exposes: {
          './AutomataCanvas': './src/components/AutomataCanvas',
          './AITutor': './src/components/AITutor',
          './SimulationEngine': './src/components/SimulationEngine'
        },
        shared: sharedDependencies
      }),
      
      // HTML generation
      new HtmlWebpackPlugin({
        template: './index.html',
        filename: 'index.html',
        inject: true,
        minify: isProduction ? {
          removeComments: true,
          collapseWhitespace: true,
          removeRedundantAttributes: true,
          useShortDoctype: true,
          removeEmptyAttributes: true,
          removeStyleLinkTypeAttributes: true,
          keepClosingSlash: true,
          minifyJS: true,
          minifyCSS: true,
          minifyURLs: true
        } : false
      }),
      
      // CSS extraction
      ...(isProduction ? [
        new MiniCssExtractPlugin({
          filename: 'assets/css/[name].[contenthash:8].css',
          chunkFilename: 'assets/css/[name].[contenthash:8].chunk.css'
        })
      ] : []),
      
      // TypeScript type checking
      new ForkTsCheckerWebpackPlugin({
        async: isDevelopment,
        typescript: {
          configFile: 'tsconfig.json'
        },
        eslint: {
          files: './src/**/*.{ts,tsx,js,jsx}'
        }
      }),
      
      // ESLint
      new ESLintPlugin({
        extensions: ['js', 'jsx', 'ts', 'tsx'],
        exclude: 'node_modules',
        context: 'src',
        cache: true,
        failOnError: !isDevelopment
      }),
      
      // Environment variables
      new webpack.DefinePlugin({
        'process.env.NODE_ENV': JSON.stringify(argv.mode || 'development'),
        '__APP_VERSION__': JSON.stringify(process.env.npm_package_version || '1.0.0'),
        '__BUILD_TIME__': JSON.stringify(new Date().toISOString())
      }),
      
      // Copy static assets
      new CopyPlugin({
        patterns: [
          {
            from: 'public',
            to: '.',
            globOptions: {
              ignore: ['**/index.html']
            }
          }
        ]
      }),
      
      // Production optimizations
      ...(isProduction ? [
        new CleanWebpackPlugin(),
        
        // Gzip compression
        new CompressionPlugin({
          algorithm: 'gzip',
          test: /\.(js|css|html|svg)$/,
          threshold: 8192,
          minRatio: 0.8
        }),
        
        // Brotli compression
        new CompressionPlugin({
          filename: '[path][base].br',
          algorithm: 'brotliCompress',
          test: /\.(js|css|html|svg)$/,
          compressionOptions: {
            level: 11
          },
          threshold: 8192,
          minRatio: 0.8
        }),
        
        // Service Worker for caching
        new WorkboxPlugin.GenerateSW({
          clientsClaim: true,
          skipWaiting: true,
          runtimeCaching: [
            {
              urlPattern: /^https:\/\/fonts\.googleapis\.com\//,
              handler: 'StaleWhileRevalidate',
              options: {
                cacheName: 'google-fonts-stylesheets'
              }
            },
            {
              urlPattern: /^https:\/\/fonts\.gstatic\.com\//,
              handler: 'CacheFirst',
              options: {
                cacheName: 'google-fonts-webfonts',
                expiration: {
                  maxEntries: 30,
                  maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
                }
              }
            }
          ]
        })
      ] : []),
      
      // Bundle analysis
      ...(isAnalyze ? [
        new BundleAnalyzerPlugin({
          analyzerMode: 'static',
          openAnalyzer: false,
          reportFilename: 'bundle-report.html'
        })
      ] : [])
    ],
    
    optimization: createOptimizationConfig(),
    
    // Development server
    devServer: isDevelopment ? createDevServerConfig() : undefined,
    
    // Source maps
    devtool: isDevelopment 
      ? 'eval-cheap-module-source-map' 
      : 'source-map',
    
    // Performance optimization
    performance: {
      maxAssetSize: 250000,
      maxEntrypointSize: 250000,
      hints: isProduction ? 'warning' : false
    },
    
    // Cache configuration for faster builds
    cache: {
      type: 'filesystem',
      cacheDirectory: path.resolve(__dirname, '.webpack-cache')
    },
    
    // Experiments
    experiments: {
      // Enable top-level await
      topLevelAwait: true,
      // Module federation v2 features
      federationRuntime: 'hoisted'
    },
    
    // Stats configuration
    stats: {
      preset: 'minimal',
      moduleTrace: true,
      errorDetails: true,
      chunks: false,
      modules: false,
      assets: false,
      entrypoints: false,
      hash: false,
      timings: true,
      builtAt: true
    },
    
    // Watch options for development
    watchOptions: {
      ignored: /node_modules/,
      aggregateTimeout: 300,
      poll: false
    }
  };
};