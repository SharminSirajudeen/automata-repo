import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Cloud, 
  Shield, 
  Smartphone, 
  CheckCircle, 
  X,
  ChevronRight,
  Lock,
  Globe,
  Zap,
  Download,
  Upload
} from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface GoogleDriveOnboardingProps {
  isOpen: boolean;
  onClose: () => void;
  onConnect: () => void;
}

export const GoogleDriveOnboarding: React.FC<GoogleDriveOnboardingProps> = ({
  isOpen,
  onClose,
  onConnect
}) => {
  const [currentStep, setCurrentStep] = useState(0);

  const features = [
    {
      icon: Cloud,
      title: "Your Progress, Your Drive",
      description: "All your learning data is stored in YOUR Google Drive. We never see or store your personal progress.",
      color: "text-blue-500"
    },
    {
      icon: Smartphone,
      title: "Learn Anywhere",
      description: "Start on your laptop, continue on your phone. Your progress syncs across all devices.",
      color: "text-green-500"
    },
    {
      icon: Shield,
      title: "Privacy First",
      description: "Your data never touches our servers. It goes directly from your browser to your Google Drive.",
      color: "text-purple-500"
    },
    {
      icon: Zap,
      title: "Instant Sync",
      description: "Changes save automatically. Never lose progress, even if your browser crashes.",
      color: "text-yellow-500"
    }
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Enhanced Header with Animations */}
              <div className="relative h-56 bg-gradient-to-br from-blue-400 via-blue-500 to-indigo-600 p-8 overflow-hidden">
                {/* Floating background elements */}
                {[...Array(15)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-2 h-2 bg-white/20 rounded-full"
                    initial={{ 
                      x: Math.random() * 400, 
                      y: Math.random() * 200,
                      scale: 0 
                    }}
                    animate={{ 
                      y: [null, -30, null],
                      scale: [0, 1, 0],
                      opacity: [0, 0.6, 0]
                    }}
                    transition={{ 
                      duration: 4 + Math.random() * 2, 
                      repeat: Infinity,
                      delay: Math.random() * 3
                    }}
                  />
                ))}

                <button
                  onClick={onClose}
                  className="absolute top-4 right-4 p-2 rounded-full bg-white/20 hover:bg-white/30 transition-all duration-200 hover:scale-110"
                >
                  <X className="w-5 h-5 text-white" />
                </button>
                
                <div className="relative z-10 flex items-center gap-6">
                  <motion.div 
                    className="p-4 bg-white rounded-3xl shadow-2xl"
                    animate={{ 
                      rotate: [0, -5, 5, 0],
                      scale: [1, 1.05, 1]
                    }}
                    transition={{ 
                      duration: 6, 
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  >
                    <svg className="w-14 h-14" viewBox="0 0 24 24">
                      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                    </svg>
                  </motion.div>
                  <div className="text-white">
                    <motion.h2 
                      className="text-4xl font-bold mb-3 drop-shadow-sm"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                    >
                      Connect Google Drive
                    </motion.h2>
                    <motion.p 
                      className="text-white/90 text-lg font-medium"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                    >
                      Transform your learning into lasting progress
                    </motion.p>
                  </div>
                </div>

                {/* Decorative cloud elements */}
                <motion.div
                  className="absolute bottom-4 right-8 opacity-30"
                  animate={{ 
                    x: [0, 10, 0],
                    y: [0, -5, 0]
                  }}
                  transition={{ 
                    duration: 8, 
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <Cloud className="w-12 h-12 text-white" />
                </motion.div>
              </div>

              {/* Content */}
              <div className="p-8">
                <AnimatePresence mode="wait">
                  {currentStep === 0 && (
                    <motion.div
                      key="features"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                    >
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                        {features.map((feature, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <Card className="p-4 border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow">
                              <div className="flex items-start gap-4">
                                <div className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-800 ${feature.color}`}>
                                  <feature.icon className="w-6 h-6" />
                                </div>
                                <div>
                                  <h3 className="font-semibold mb-1">{feature.title}</h3>
                                  <p className="text-sm text-gray-600 dark:text-gray-400">
                                    {feature.description}
                                  </p>
                                </div>
                              </div>
                            </Card>
                          </motion.div>
                        ))}
                      </div>

                      <div className="flex items-center justify-between">
                        <Button
                          variant="ghost"
                          onClick={onClose}
                          className="text-gray-600"
                        >
                          Continue without saving
                        </Button>
                        <Button
                          onClick={() => setCurrentStep(1)}
                          className="bg-blue-600 hover:bg-blue-700 text-white"
                        >
                          Learn More
                          <ChevronRight className="w-4 h-4 ml-2" />
                        </Button>
                      </div>
                    </motion.div>
                  )}

                  {currentStep === 1 && (
                    <motion.div
                      key="privacy"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                    >
                      <div className="mb-8">
                        <h3 className="text-2xl font-bold mb-4">Your Privacy Matters</h3>
                        
                        <div className="space-y-4">
                          <div className="flex items-start gap-4">
                            <Lock className="w-5 h-5 text-green-500 mt-1" />
                            <div>
                              <h4 className="font-semibold mb-1">End-to-End Privacy</h4>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                Your progress data travels directly from your browser to your Google Drive. 
                                We don't have servers that store or process your personal data.
                              </p>
                            </div>
                          </div>

                          <div className="flex items-start gap-4">
                            <Globe className="w-5 h-5 text-blue-500 mt-1" />
                            <div>
                              <h4 className="font-semibold mb-1">You Own Your Data</h4>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                All files are stored in a folder called "Automata Progress" in your Drive. 
                                You can view, download, or delete them anytime.
                              </p>
                            </div>
                          </div>

                          <div className="flex items-start gap-4">
                            <Download className="w-5 h-5 text-purple-500 mt-1" />
                            <div>
                              <h4 className="font-semibold mb-1">Export Anytime</h4>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                Download your progress as a JSON file. Import it on another device or keep it as a backup.
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                          <p className="text-sm text-blue-800 dark:text-blue-200">
                            <strong>Note:</strong> This app only requests permission to create and manage its own files. 
                            It cannot access any other files in your Google Drive.
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <Button
                          variant="ghost"
                          onClick={() => setCurrentStep(0)}
                        >
                          Back
                        </Button>
                        <Button
                          onClick={onConnect}
                          className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white"
                        >
                          <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                            <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                            <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                            <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                            <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                          </svg>
                          Connect Google Drive
                        </Button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};