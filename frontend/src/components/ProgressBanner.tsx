import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useSpring, useMotionValue } from 'framer-motion';
import { 
  Cloud, 
  CloudOff, 
  CheckCircle, 
  Save, 
  TrendingUp, 
  Trophy, 
  Sparkles, 
  Star,
  Zap,
  Heart,
  Award,
  Target,
  Calendar,
  Clock,
  Flame
} from 'lucide-react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Card } from './ui/card';
import { cn } from '@/lib/utils';

interface ProgressBannerProps {
  completedProblems: number;
  totalProblems: number;
  streak: number;
  isSignedIn: boolean;
  onSaveToGoogle: () => void;
  timeSpentToday?: number; // in minutes
  currentLevel?: number;
  xpGained?: number;
}

export const ProgressBanner: React.FC<ProgressBannerProps> = ({
  completedProblems,
  totalProblems,
  streak,
  isSignedIn,
  onSaveToGoogle,
  timeSpentToday = 0,
  currentLevel = 1,
  xpGained = 0
}) => {
  const [showSavePrompt, setShowSavePrompt] = useState(false);
  const [showMilestoneModal, setShowMilestoneModal] = useState(false);
  const [lastCelebrated, setLastCelebrated] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const progressPercentage = (completedProblems / totalProblems) * 100;
  const streakIconRef = useRef<HTMLDivElement>(null);
  
  // Enhanced milestone detection
  const getMilestoneType = (problems: number) => {
    if (problems % 50 === 0) return 'legendary';
    if (problems % 25 === 0) return 'epic';
    if (problems % 10 === 0) return 'major';
    if (problems % 5 === 0) return 'minor';
    return null;
  };

  // Show save prompt after completing 3 problems or hitting milestones
  useEffect(() => {
    if ((completedProblems >= 3 || getMilestoneType(completedProblems)) && !isSignedIn) {
      setShowSavePrompt(true);
    }
  }, [completedProblems, isSignedIn]);

  // Milestone celebration logic
  useEffect(() => {
    const milestone = getMilestoneType(completedProblems);
    if (milestone && completedProblems > lastCelebrated) {
      setShowMilestoneModal(true);
      setLastCelebrated(completedProblems);
      setTimeout(() => setShowMilestoneModal(false), 6000);
    }
  }, [completedProblems, lastCelebrated]);

  // Animate streak flame when it increases
  useEffect(() => {
    if (streak > 0) {
      setIsAnimating(true);
      setTimeout(() => setIsAnimating(false), 1000);
    }
  }, [streak]);

  return (
    <>
      {/* Enhanced Floating Progress Dashboard */}
      <motion.div
        className="fixed top-4 right-4 z-50"
        initial={{ opacity: 0, x: 100, scale: 0.8 }}
        animate={{ opacity: 1, x: 0, scale: 1 }}
        transition={{ type: "spring", damping: 20, stiffness: 300 }}
        whileHover={{ scale: 1.02 }}
      >
        <Card className="p-5 bg-white/98 dark:bg-gray-900/98 backdrop-blur-xl shadow-2xl border-0 rounded-2xl overflow-hidden">
          {/* Gradient Background */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 via-purple-50/30 to-pink-50/20 dark:from-blue-950/50 dark:via-purple-950/30 dark:to-pink-950/20" />
          
          <div className="relative z-10 flex items-center gap-5">
            {/* Enhanced Progress Ring with Sparkles */}
            <div className="relative w-20 h-20">
              {/* Outer glow ring */}
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 opacity-20 blur-sm" />
              
              <svg className="transform -rotate-90 w-20 h-20 relative z-10">
                <circle
                  cx="40"
                  cy="40"
                  r="34"
                  stroke="currentColor"
                  strokeWidth="3"
                  fill="none"
                  className="text-gray-200 dark:text-gray-700"
                />
                <motion.circle
                  cx="40"
                  cy="40"
                  r="34"
                  stroke="url(#progressGradient)"
                  strokeWidth="3"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 34}`}
                  strokeDashoffset={`${2 * Math.PI * 34 * (1 - progressPercentage / 100)}`}
                  className="drop-shadow-sm"
                  initial={{ strokeDashoffset: `${2 * Math.PI * 34}` }}
                  animate={{ strokeDashoffset: `${2 * Math.PI * 34 * (1 - progressPercentage / 100)}` }}
                  transition={{ duration: 2, ease: "easeOut" }}
                />
                <defs>
                  <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#3B82F6" />
                    <stop offset="50%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#EC4899" />
                  </linearGradient>
                </defs>
              </svg>
              
              {/* Center content with level badge */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    {completedProblems}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                    L{currentLevel}
                  </div>
                </div>
              </div>

              {/* Floating sparkles */}
              {progressPercentage > 50 && (
                <motion.div
                  className="absolute -top-1 -right-1"
                  animate={{ 
                    rotate: [0, 360],
                    scale: [0.8, 1.2, 0.8]
                  }}
                  transition={{ 
                    duration: 3, 
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <Sparkles className="w-4 h-4 text-yellow-500" />
                </motion.div>
              )}
            </div>
            
            {/* Enhanced Stats Grid */}
            <div className="flex flex-col gap-3 min-w-[140px]">
              {/* Streak with flame animation */}
              <motion.div 
                className="flex items-center gap-2"
                animate={isAnimating ? { scale: [1, 1.1, 1] } : {}}
                transition={{ duration: 0.5 }}
              >
                <motion.div
                  ref={streakIconRef}
                  animate={streak > 5 ? { 
                    rotate: [-5, 5, -5],
                  } : {}}
                  transition={{ 
                    duration: 0.8, 
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <Flame className={cn(
                    "w-4 h-4",
                    streak === 0 && "text-gray-400",
                    streak > 0 && streak < 3 && "text-orange-400",
                    streak >= 3 && streak < 7 && "text-orange-500",
                    streak >= 7 && streak < 15 && "text-red-500",
                    streak >= 15 && "text-red-600"
                  )} />
                </motion.div>
                <div className="flex flex-col">
                  <span className="text-sm font-semibold">{streak} day streak</span>
                  {streak >= 7 && (
                    <span className="text-xs text-orange-600 font-medium">🔥 On fire!</span>
                  )}
                </div>
              </motion.div>

              {/* Time spent today */}
              {timeSpentToday > 0 && (
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-green-500" />
                  <span className="text-sm font-medium">{timeSpentToday}min today</span>
                </div>
              )}

              {/* Sync status with enhanced visual feedback */}
              <div className="flex items-center gap-2">
                {isSignedIn ? (
                  <>
                    <motion.div
                      animate={{ 
                        scale: [1, 1.2, 1],
                        opacity: [0.7, 1, 0.7]
                      }}
                      transition={{ 
                        duration: 2, 
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    >
                      <Cloud className="w-4 h-4 text-blue-500" />
                    </motion.div>
                    <div className="flex flex-col">
                      <span className="text-xs font-medium text-blue-600">Synced</span>
                      <span className="text-xs text-gray-500">Auto-saving</span>
                    </div>
                  </>
                ) : (
                  <>
                    <CloudOff className="w-4 h-4 text-amber-500" />
                    <div className="flex flex-col">
                      <span className="text-xs font-medium text-amber-600">Local only</span>
                      <span className="text-xs text-gray-500">Tap to sync</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Quick action button for non-signed users */}
          {!isSignedIn && (
            <motion.div
              className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              transition={{ delay: 1 }}
            >
              <Button
                size="sm"
                onClick={onSaveToGoogle}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white border-0 text-xs font-medium"
              >
                <Cloud className="w-3 h-3 mr-1" />
                Sync Progress
              </Button>
            </motion.div>
          )}
        </Card>
      </motion.div>

      {/* Enhanced Save Progress Prompt */}
      <AnimatePresence>
        {showSavePrompt && !isSignedIn && (
          <motion.div
            className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50 max-w-lg w-full px-4"
            initial={{ opacity: 0, y: 100, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 100, scale: 0.95 }}
            transition={{ type: "spring", damping: 25, stiffness: 400 }}
          >
            <Card className="relative p-6 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-blue-950/90 dark:via-indigo-950/90 dark:to-purple-950/90 border-blue-200 dark:border-blue-800 shadow-2xl backdrop-blur-sm overflow-hidden">
              {/* Animated background particles */}
              <div className="absolute inset-0 overflow-hidden">
                {[...Array(12)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-1 h-1 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full opacity-30"
                    initial={{ 
                      x: Math.random() * 400, 
                      y: Math.random() * 200,
                      scale: 0 
                    }}
                    animate={{ 
                      y: [null, -20, null],
                      scale: [0, 1, 0],
                      opacity: [0, 0.6, 0]
                    }}
                    transition={{ 
                      duration: 3 + Math.random() * 2, 
                      repeat: Infinity,
                      delay: Math.random() * 2
                    }}
                  />
                ))}
              </div>

              <div className="relative z-10 flex items-start gap-4">
                <motion.div 
                  className="p-4 bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900 dark:to-purple-900 rounded-2xl"
                  animate={{ 
                    rotate: [0, -10, 10, 0],
                    scale: [1, 1.1, 1]
                  }}
                  transition={{ 
                    duration: 4, 
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <Heart className="w-7 h-7 text-blue-600 dark:text-blue-400" />
                </motion.div>
                
                <div className="flex-1">
                  <motion.h3 
                    className="font-bold text-xl mb-2 bg-gradient-to-r from-blue-700 to-purple-700 dark:from-blue-300 dark:to-purple-300 bg-clip-text text-transparent"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    Your progress deserves protection! 
                  </motion.h3>
                  
                  <motion.p 
                    className="text-sm text-gray-700 dark:text-gray-300 mb-5 leading-relaxed"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    You've solved <strong>{completedProblems} problems</strong> and built a <strong>{streak}-day streak</strong>. 
                    Keep this momentum safe in your Google Drive:
                  </motion.p>
                  
                  <motion.div 
                    className="grid grid-cols-1 gap-3 mb-5"
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    {[
                      { icon: Target, text: "Continue your streak anywhere", color: "text-green-500" },
                      { icon: Award, text: "Never lose achievements", color: "text-yellow-500" },
                      { icon: Zap, text: "Instant sync across devices", color: "text-purple-500" }
                    ].map((item, index) => (
                      <motion.li 
                        key={index}
                        className="flex items-center gap-3 text-sm p-2 rounded-lg bg-white/60 dark:bg-gray-800/40"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 + index * 0.1 }}
                      >
                        <item.icon className={`w-4 h-4 ${item.color}`} />
                        <span className="font-medium">{item.text}</span>
                      </motion.li>
                    ))}
                  </motion.div>
                  
                  <motion.div 
                    className="flex gap-3"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 }}
                  >
                    <Button
                      onClick={onSaveToGoogle}
                      className="flex-1 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-700 hover:from-blue-700 hover:via-purple-700 hover:to-blue-800 text-white shadow-lg transition-all duration-300 transform hover:scale-[1.02]"
                    >
                      <motion.div
                        animate={{ rotate: [0, 360] }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      >
                        <Star className="w-4 h-4 mr-2" />
                      </motion.div>
                      Secure My Progress
                    </Button>
                    <Button
                      variant="ghost"
                      onClick={() => setShowSavePrompt(false)}
                      className="text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
                    >
                      Later
                    </Button>
                  </motion.div>
                </div>
              </div>
              
              {/* Enhanced Trust Badge */}
              <motion.div 
                className="mt-5 pt-4 border-t border-blue-200/60 dark:border-blue-800/60"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 }}
              >
                <div className="flex items-center justify-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                  <svg className="w-4 h-4 text-green-500" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
                    <path d="M10 17l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z" fill="white"/>
                  </svg>
                  <span className="font-medium">100% Private:</span>
                  <span>Your data goes directly to YOUR Google Drive. We never see it.</span>
                </div>
              </motion.div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Epic Milestone Celebrations */}
      <AnimatePresence>
        {showMilestoneModal && (
          <>
            {/* Celebration Backdrop */}
            <motion.div
              className="fixed inset-0 bg-gradient-to-br from-yellow-900/20 via-orange-900/20 to-red-900/20 backdrop-blur-sm z-50"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {/* Confetti particles */}
              {[...Array(50)].map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: ['#FFD700', '#FF6B35', '#F7931E', '#FF1744', '#9C27B0'][i % 5],
                    left: `${Math.random() * 100}%`,
                    top: `${Math.random() * 100}%`,
                  }}
                  initial={{ scale: 0, y: 0, rotate: 0 }}
                  animate={{ 
                    scale: [0, 1, 0],
                    y: [0, -100, 100],
                    rotate: [0, 180, 360],
                    opacity: [0, 1, 0]
                  }}
                  transition={{ 
                    duration: 3 + Math.random() * 2,
                    delay: Math.random() * 2,
                    ease: "easeOut"
                  }}
                />
              ))}
            </motion.div>

            {/* Main Celebration Modal */}
            <motion.div
              className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50"
              initial={{ scale: 0, opacity: 0, rotateY: 180 }}
              animate={{ scale: 1, opacity: 1, rotateY: 0 }}
              exit={{ scale: 0, opacity: 0, rotateY: -180 }}
              transition={{ type: "spring", damping: 15, stiffness: 200 }}
            >
              <Card className="relative p-10 bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50 dark:from-yellow-950/95 dark:via-orange-950/95 dark:to-red-950/95 shadow-2xl max-w-md border-2 border-yellow-200 dark:border-yellow-800 overflow-hidden">
                {/* Animated border glow */}
                <div className="absolute inset-0 bg-gradient-to-r from-yellow-400 via-orange-400 to-red-400 rounded-lg blur-sm opacity-30 animate-pulse" />
                
                <div className="relative z-10 text-center">
                  {/* Dynamic Trophy Animation */}
                  <motion.div
                    className="relative mb-6"
                    animate={{ 
                      rotate: [0, -10, 10, 0],
                      scale: [1, 1.1, 1]
                    }}
                    transition={{ 
                      duration: 2, 
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  >
                    <div className="relative">
                      <Trophy className="w-20 h-20 text-yellow-500 mx-auto drop-shadow-lg" />
                      {/* Radiating lines */}
                      {[...Array(8)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="absolute w-1 h-8 bg-gradient-to-t from-yellow-400 to-transparent rounded-full"
                          style={{
                            left: '50%',
                            top: '50%',
                            transformOrigin: '50% 100%',
                            transform: `translate(-50%, -100%) rotate(${i * 45}deg)`
                          }}
                          animate={{ 
                            opacity: [0.3, 1, 0.3],
                            scaleY: [0.5, 1, 0.5]
                          }}
                          transition={{ 
                            duration: 2, 
                            repeat: Infinity,
                            delay: i * 0.1,
                            ease: "easeInOut"
                          }}
                        />
                      ))}
                    </div>
                  </motion.div>

                  {/* Dynamic Title Based on Milestone */}
                  <motion.h2 
                    className="text-3xl font-bold mb-3 bg-gradient-to-r from-yellow-600 via-orange-600 to-red-600 bg-clip-text text-transparent"
                    initial={{ scale: 0.5 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.3, type: "spring", damping: 10 }}
                  >
                    {getMilestoneType(completedProblems) === 'legendary' && "🌟 LEGENDARY ACHIEVEMENT!"}
                    {getMilestoneType(completedProblems) === 'epic' && "⚡ EPIC MILESTONE!"}
                    {getMilestoneType(completedProblems) === 'major' && "🎯 MAJOR MILESTONE!"}
                    {getMilestoneType(completedProblems) === 'minor' && "✨ GREAT PROGRESS!"}
                  </motion.h2>
                  
                  <motion.div
                    className="space-y-2 mb-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                  >
                    <p className="text-xl font-semibold text-gray-800 dark:text-gray-200">
                      {completedProblems} Problems Conquered!
                    </p>
                    {streak > 0 && (
                      <p className="text-lg text-orange-600 dark:text-orange-400 font-medium">
                        🔥 {streak} Day Learning Streak
                      </p>
                    )}
                    {timeSpentToday > 0 && (
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {timeSpentToday} minutes of focused learning today
                      </p>
                    )}
                  </motion.div>

                  {/* Motivational Message */}
                  <motion.p 
                    className="text-sm text-gray-700 dark:text-gray-300 mb-6 leading-relaxed"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.7 }}
                  >
                    {completedProblems >= 50 && "You're a theory of computation master! Your dedication is inspiring."}
                    {completedProblems >= 25 && completedProblems < 50 && "Incredible progress! You're building serious expertise."}
                    {completedProblems >= 10 && completedProblems < 25 && "Amazing momentum! You're on the path to mastery."}
                    {completedProblems < 10 && "Great start! Every expert was once a beginner."}
                  </motion.p>

                  {/* Save to Google Call-to-Action */}
                  {!isSignedIn && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.9 }}
                    >
                      <Button 
                        onClick={onSaveToGoogle} 
                        className="bg-gradient-to-r from-yellow-600 via-orange-600 to-red-600 hover:from-yellow-700 hover:via-orange-700 hover:to-red-700 text-white shadow-lg transform hover:scale-105 transition-all duration-300"
                      >
                        <Trophy className="w-4 h-4 mr-2" />
                        Preserve This Achievement Forever
                      </Button>
                    </motion.div>
                  )}

                  {/* Celebration Timer */}
                  <motion.div
                    className="mt-4 text-xs text-gray-500 dark:text-gray-400"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1.2 }}
                  >
                    🎉 Celebrating your achievement...
                  </motion.div>
                </div>
              </Card>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};