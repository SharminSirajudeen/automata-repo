import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle2, 
  Heart, 
  Sparkles, 
  Zap, 
  Star,
  Trophy,
  Target,
  Award,
  Flame
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Success Animation Component
interface SuccessAnimationProps {
  isVisible: boolean;
  onComplete?: () => void;
  type?: 'problem' | 'streak' | 'milestone' | 'sync';
}

export const SuccessAnimation: React.FC<SuccessAnimationProps> = ({ 
  isVisible, 
  onComplete,
  type = 'problem'
}) => {
  useEffect(() => {
    if (isVisible && onComplete) {
      const timer = setTimeout(onComplete, 2000);
      return () => clearTimeout(timer);
    }
  }, [isVisible, onComplete]);

  const getIcon = () => {
    switch (type) {
      case 'streak': return Flame;
      case 'milestone': return Trophy;
      case 'sync': return Zap;
      default: return CheckCircle2;
    }
  };

  const getColor = () => {
    switch (type) {
      case 'streak': return 'text-orange-500';
      case 'milestone': return 'text-yellow-500';
      case 'sync': return 'text-blue-500';
      default: return 'text-green-500';
    }
  };

  const Icon = getIcon();

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 pointer-events-none"
          initial={{ scale: 0, opacity: 0, rotate: -180 }}
          animate={{ scale: 1, opacity: 1, rotate: 0 }}
          exit={{ scale: 0, opacity: 0, rotate: 180 }}
          transition={{ type: "spring", damping: 15, stiffness: 300 }}
        >
          <div className="relative">
            {/* Main icon */}
            <motion.div
              animate={{ 
                scale: [1, 1.2, 1],
                rotate: [0, 10, -10, 0]
              }}
              transition={{ 
                duration: 0.6,
                repeat: 1,
                delay: 0.2
              }}
            >
              <Icon className={cn("w-16 h-16", getColor())} />
            </motion.div>

            {/* Radiating particles */}
            {[...Array(8)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 bg-current rounded-full"
                style={{
                  left: '50%',
                  top: '50%',
                  color: getColor().replace('text-', '').replace('-500', '')
                }}
                initial={{ 
                  scale: 0,
                  x: '-50%',
                  y: '-50%'
                }}
                animate={{ 
                  scale: [0, 1, 0],
                  x: ['-50%', `${Math.cos(i * Math.PI / 4) * 40 - 50}%`],
                  y: ['-50%', `${Math.sin(i * Math.PI / 4) * 40 - 50}%`]
                }}
                transition={{ 
                  duration: 1,
                  delay: 0.5 + i * 0.1,
                  ease: "easeOut"
                }}
              />
            ))}

            {/* Ripple effect */}
            <motion.div
              className="absolute inset-0 border-2 rounded-full"
              style={{ borderColor: getColor().replace('text-', '').replace('-500', '') }}
              initial={{ scale: 0, opacity: 0.8 }}
              animate={{ scale: 3, opacity: 0 }}
              transition={{ duration: 1, delay: 0.3 }}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Floating Heart Animation for positive feedback
interface FloatingHeartsProps {
  isActive: boolean;
  count?: number;
}

export const FloatingHearts: React.FC<FloatingHeartsProps> = ({ 
  isActive, 
  count = 5 
}) => {
  return (
    <AnimatePresence>
      {isActive && (
        <div className="fixed inset-0 pointer-events-none z-40">
          {[...Array(count)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute"
              style={{
                left: `${20 + Math.random() * 60}%`,
                top: `${20 + Math.random() * 60}%`,
              }}
              initial={{ 
                scale: 0,
                opacity: 0,
                y: 0,
                rotate: 0
              }}
              animate={{ 
                scale: [0, 1, 0],
                opacity: [0, 1, 0],
                y: [-100, -200],
                rotate: [0, 360],
                x: [0, Math.random() * 100 - 50]
              }}
              transition={{ 
                duration: 2 + Math.random(),
                delay: i * 0.2,
                ease: "easeOut"
              }}
            >
              <Heart className="w-6 h-6 text-pink-500 fill-pink-500" />
            </motion.div>
          ))}
        </div>
      )}
    </AnimatePresence>
  );
};

// Sparkle Trail for cursor interactions
interface SparkleTrailProps {
  isActive: boolean;
}

export const SparkleTrail: React.FC<SparkleTrailProps> = ({ isActive }) => {
  const [sparkles, setSparkles] = useState<Array<{ id: number; x: number; y: number }>>([]);

  useEffect(() => {
    if (!isActive) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newSparkle = {
        id: Date.now(),
        x: e.clientX,
        y: e.clientY
      };
      
      setSparkles(prev => [...prev.slice(-5), newSparkle]);
    };

    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, [isActive]);

  return (
    <div className="fixed inset-0 pointer-events-none z-30">
      {sparkles.map(sparkle => (
        <motion.div
          key={sparkle.id}
          className="absolute"
          style={{ left: sparkle.x, top: sparkle.y }}
          initial={{ scale: 0, opacity: 1 }}
          animate={{ scale: 1, opacity: 0 }}
          exit={{ scale: 0 }}
          transition={{ duration: 0.8 }}
        >
          <Sparkles className="w-4 h-4 text-yellow-400 -translate-x-2 -translate-y-2" />
        </motion.div>
      ))}
    </div>
  );
};

// Progress Celebration Burst
interface ProgressBurstProps {
  isVisible: boolean;
  progress: number; // 0-100
  onComplete?: () => void;
}

export const ProgressBurst: React.FC<ProgressBurstProps> = ({ 
  isVisible, 
  progress,
  onComplete 
}) => {
  useEffect(() => {
    if (isVisible && onComplete) {
      const timer = setTimeout(onComplete, 3000);
      return () => clearTimeout(timer);
    }
  }, [isVisible, onComplete]);

  const particleCount = Math.min(Math.floor(progress / 5), 20);

  return (
    <AnimatePresence>
      {isVisible && (
        <div className="fixed inset-0 pointer-events-none z-40">
          {[...Array(particleCount)].map((_, i) => {
            const angle = (i / particleCount) * 2 * Math.PI;
            const distance = 100 + Math.random() * 200;
            const x = Math.cos(angle) * distance;
            const y = Math.sin(angle) * distance;
            
            return (
              <motion.div
                key={i}
                className="absolute top-1/2 left-1/2"
                initial={{ 
                  x: 0, 
                  y: 0, 
                  scale: 0,
                  opacity: 1
                }}
                animate={{ 
                  x: x, 
                  y: y, 
                  scale: [0, 1, 0],
                  opacity: [1, 1, 0],
                  rotate: [0, 360]
                }}
                transition={{ 
                  duration: 2,
                  delay: i * 0.05,
                  ease: "easeOut"
                }}
              >
                <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />
              </motion.div>
            );
          })}
          
          {/* Central burst */}
          <motion.div
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: [0, 2, 1], opacity: [0, 1, 0.8] }}
            transition={{ duration: 1 }}
          >
            <div className="relative">
              <Trophy className="w-16 h-16 text-yellow-500" />
              <motion.div
                className="absolute inset-0 border-4 border-yellow-400 rounded-full"
                animate={{ scale: [1, 2, 3], opacity: [0.8, 0.4, 0] }}
                transition={{ duration: 1.5, repeat: 2 }}
              />
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

// Motivational Quote Popup
interface MotivationalQuoteProps {
  isVisible: boolean;
  streak: number;
  completedProblems: number;
  onClose?: () => void;
}

export const MotivationalQuote: React.FC<MotivationalQuoteProps> = ({
  isVisible,
  streak,
  completedProblems,
  onClose
}) => {
  const getQuote = () => {
    if (streak >= 30) return "Consistency is the mother of mastery! 🚀";
    if (streak >= 14) return "Two weeks strong! You're building an unstoppable habit! 💪";
    if (streak >= 7) return "One week of dedication! Excellence is becoming your standard! ⭐";
    if (completedProblems >= 50) return "50 problems conquered! You're a theory of computation warrior! 🛡️";
    if (completedProblems >= 25) return "Quarter century of problems solved! Your expertise is growing! 📈";
    if (completedProblems >= 10) return "Double digits! You're gaining serious momentum! 🎯";
    return "Every problem solved makes you stronger! Keep going! 💎";
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          className="fixed bottom-8 right-8 z-50 max-w-sm"
          initial={{ opacity: 0, x: 100, rotate: 5 }}
          animate={{ opacity: 1, x: 0, rotate: 0 }}
          exit={{ opacity: 0, x: 100, rotate: -5 }}
          transition={{ type: "spring", damping: 20 }}
        >
          <div className="bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 p-[2px] rounded-xl">
            <div className="bg-white dark:bg-gray-900 rounded-xl p-4 relative">
              <motion.div
                className="absolute -top-2 -left-2"
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
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </motion.div>
              
              <p className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-3">
                {getQuote()}
              </p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-indigo-500" />
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    Keep pushing forward!
                  </span>
                </div>
                
                {onClose && (
                  <button
                    onClick={onClose}
                    className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Sync Status Indicator
interface SyncStatusProps {
  status: 'idle' | 'syncing' | 'success' | 'error';
  message?: string;
}

export const SyncStatus: React.FC<SyncStatusProps> = ({ status, message }) => {
  return (
    <AnimatePresence>
      {status !== 'idle' && (
        <motion.div
          className="fixed top-8 left-1/2 transform -translate-x-1/2 z-50"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -50 }}
          transition={{ type: "spring", damping: 20 }}
        >
          <div className={cn(
            "px-4 py-2 rounded-full shadow-lg text-sm font-medium flex items-center gap-2",
            status === 'syncing' && "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
            status === 'success' && "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
            status === 'error' && "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
          )}>
            {status === 'syncing' && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Zap className="w-4 h-4" />
              </motion.div>
            )}
            {status === 'success' && <CheckCircle2 className="w-4 h-4" />}
            {status === 'error' && <span className="w-4 h-4">⚠️</span>}
            
            {message || (
              <>
                {status === 'syncing' && 'Syncing to Google Drive...'}
                {status === 'success' && 'Progress saved successfully!'}
                {status === 'error' && 'Sync failed. Trying again...'}
              </>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Streak Fire Animation
interface StreakFireProps {
  streak: number;
  isActive: boolean;
}

export const StreakFire: React.FC<StreakFireProps> = ({ streak, isActive }) => {
  if (streak === 0 || !isActive) return null;

  const intensity = Math.min(streak / 30, 1); // Max intensity at 30 days
  const fireCount = Math.floor(3 + intensity * 5); // 3-8 flames

  return (
    <div className="relative inline-block">
      {[...Array(fireCount)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute"
          style={{
            left: `${-10 + i * 5}px`,
            bottom: '100%'
          }}
          animate={{
            y: [0, -20, 0],
            opacity: [0.3, 1, 0.3],
            scale: [0.8, 1.2, 0.8]
          }}
          transition={{
            duration: 1 + Math.random() * 0.5,
            repeat: Infinity,
            delay: i * 0.1,
            ease: "easeInOut"
          }}
        >
          <div className={cn(
            "w-2 h-4 rounded-full",
            streak < 7 && "bg-orange-400",
            streak >= 7 && streak < 15 && "bg-red-500",
            streak >= 15 && "bg-red-600"
          )} />
        </motion.div>
      ))}
    </div>
  );
};