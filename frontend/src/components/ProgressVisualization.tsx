import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, 
  Calendar, 
  Target, 
  Clock, 
  Award, 
  Flame,
  Star,
  Brain,
  Zap,
  BarChart3
} from 'lucide-react';
import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { cn } from '@/lib/utils';

interface ProgressData {
  completedProblems: number;
  totalProblems: number;
  streak: number;
  longestStreak: number;
  timeSpentToday: number; // in minutes
  timeSpentTotal: number; // in hours
  currentLevel: number;
  xpGained: number;
  problemsThisWeek: number[];
  achievements: Achievement[];
  accuracyRate: number;
  averageTime: number; // minutes per problem
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  unlockedAt: string;
}

interface ProgressVisualizationProps {
  progress: ProgressData;
  isSignedIn: boolean;
  onSaveToGoogle?: () => void;
}

export const ProgressVisualization: React.FC<ProgressVisualizationProps> = ({
  progress,
  isSignedIn,
  onSaveToGoogle
}) => {
  const [selectedMetric, setSelectedMetric] = useState<'overview' | 'streak' | 'time' | 'achievements'>('overview');
  const [isAnimating, setIsAnimating] = useState(false);

  const progressPercentage = (progress.completedProblems / progress.totalProblems) * 100;
  const xpToNextLevel = 100; // Simplified XP system
  const currentLevelProgress = (progress.xpGained % xpToNextLevel) / xpToNextLevel * 100;

  // Calculate learning velocity (problems per day over last week)
  const learningVelocity = progress.problemsThisWeek.reduce((sum, count) => sum + count, 0) / 7;

  useEffect(() => {
    setIsAnimating(true);
    const timer = setTimeout(() => setIsAnimating(false), 1000);
    return () => clearTimeout(timer);
  }, [progress.completedProblems]);

  const renderOverview = () => (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Main Progress Ring */}
      <div className="flex items-center justify-center">
        <div className="relative w-40 h-40">
          {/* Outer glow */}
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 opacity-20 blur-md animate-pulse" />
          
          <svg className="transform -rotate-90 w-40 h-40 relative z-10">
            <circle
              cx="80"
              cy="80"
              r="70"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
              className="text-gray-200 dark:text-gray-700"
            />
            <motion.circle
              cx="80"
              cy="80"
              r="70"
              stroke="url(#mainProgressGradient)"
              strokeWidth="4"
              fill="none"
              strokeDasharray={`${2 * Math.PI * 70}`}
              strokeLinecap="round"
              initial={{ strokeDashoffset: `${2 * Math.PI * 70}` }}
              animate={{ strokeDashoffset: `${2 * Math.PI * 70 * (1 - progressPercentage / 100)}` }}
              transition={{ duration: 2, ease: "easeOut" }}
            />
            <defs>
              <linearGradient id="mainProgressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#3B82F6" />
                <stop offset="50%" stopColor="#8B5CF6" />
                <stop offset="100%" stopColor="#EC4899" />
              </linearGradient>
            </defs>
          </svg>
          
          {/* Center content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.div 
              className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
              animate={isAnimating ? { scale: [1, 1.1, 1] } : {}}
              transition={{ duration: 0.5 }}
            >
              {progress.completedProblems}
            </motion.div>
            <div className="text-sm text-gray-500 dark:text-gray-400 font-medium">
              of {progress.totalProblems} problems
            </div>
            <div className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              {progressPercentage.toFixed(1)}% complete
            </div>
          </div>

          {/* Floating elements */}
          {progressPercentage > 25 && (
            <motion.div
              className="absolute -top-2 -right-2"
              animate={{ 
                rotate: [0, 360],
                scale: [0.8, 1.2, 0.8]
              }}
              transition={{ 
                duration: 4, 
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <Star className="w-6 h-6 text-yellow-500" />
            </motion.div>
          )}
        </div>
      </div>

      {/* Level Progress */}
      <Card className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950 dark:to-purple-950 border-indigo-200 dark:border-indigo-800">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-indigo-100 dark:bg-indigo-900 rounded-full">
            <Brain className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-indigo-900 dark:text-indigo-100">
                Level {progress.currentLevel}
              </span>
              <span className="text-sm text-indigo-600 dark:text-indigo-400">
                {progress.xpGained % xpToNextLevel}/{xpToNextLevel} XP
              </span>
            </div>
            <Progress 
              value={currentLevelProgress} 
              className="h-2 bg-indigo-200 dark:bg-indigo-800"
            />
          </div>
        </div>
      </Card>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 gap-4">
        <motion.div
          whileHover={{ scale: 1.02 }}
          onClick={() => setSelectedMetric('streak')}
          className="cursor-pointer"
        >
          <Card className="p-4 hover:shadow-lg transition-all duration-200">
            <div className="flex items-center gap-3">
              <Flame className={cn(
                "w-8 h-8",
                progress.streak === 0 && "text-gray-400",
                progress.streak > 0 && progress.streak < 3 && "text-orange-400",
                progress.streak >= 3 && progress.streak < 7 && "text-orange-500",
                progress.streak >= 7 && "text-red-500"
              )} />
              <div>
                <div className="text-2xl font-bold">{progress.streak}</div>
                <div className="text-sm text-gray-500">Day Streak</div>
              </div>
            </div>
          </Card>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.02 }}
          onClick={() => setSelectedMetric('time')}
          className="cursor-pointer"
        >
          <Card className="p-4 hover:shadow-lg transition-all duration-200">
            <div className="flex items-center gap-3">
              <Clock className="w-8 h-8 text-green-500" />
              <div>
                <div className="text-2xl font-bold">{progress.timeSpentToday}</div>
                <div className="text-sm text-gray-500">Min Today</div>
              </div>
            </div>
          </Card>
        </motion.div>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <Target className="w-8 h-8 text-blue-500" />
            <div>
              <div className="text-2xl font-bold">{progress.accuracyRate}%</div>
              <div className="text-sm text-gray-500">Accuracy</div>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-purple-500" />
            <div>
              <div className="text-2xl font-bold">{learningVelocity.toFixed(1)}</div>
              <div className="text-sm text-gray-500">Per Day</div>
            </div>
          </div>
        </Card>
      </div>
    </motion.div>
  );

  const renderStreak = () => (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center">
        <motion.div
          animate={progress.streak > 5 ? { 
            rotate: [-5, 5, -5],
            scale: [1, 1.1, 1]
          } : {}}
          transition={{ 
            duration: 1, 
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <Flame className="w-24 h-24 mx-auto text-orange-500 mb-4" />
        </motion.div>
        <h3 className="text-3xl font-bold mb-2">{progress.streak} Day Streak</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          {progress.streak === 0 && "Start your learning journey today!"}
          {progress.streak > 0 && progress.streak < 3 && "Great start! Keep the momentum going."}
          {progress.streak >= 3 && progress.streak < 7 && "Building consistency! You're on fire."}
          {progress.streak >= 7 && progress.streak < 15 && "Incredible dedication! You're unstoppable."}
          {progress.streak >= 15 && "Legendary streak! You're a learning machine."}
        </p>
      </div>

      <Card className="p-6 bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950">
        <div className="flex items-center justify-between mb-4">
          <span className="font-semibold">Personal Best</span>
          <span className="text-2xl font-bold text-orange-600">{progress.longestStreak} days</span>
        </div>
        <Progress 
          value={(progress.streak / Math.max(progress.longestStreak, progress.streak)) * 100} 
          className="h-3"
        />
      </Card>

      {/* Week visualization */}
      <div>
        <h4 className="font-semibold mb-3">This Week's Progress</h4>
        <div className="grid grid-cols-7 gap-2">
          {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((day, index) => (
            <div key={index} className="text-center">
              <div className="text-xs text-gray-500 mb-2">{day}</div>
              <motion.div
                className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center font-bold text-sm",
                  progress.problemsThisWeek[index] > 0 
                    ? "bg-gradient-to-br from-orange-400 to-red-500 text-white shadow-lg"
                    : "bg-gray-100 dark:bg-gray-800 text-gray-400"
                )}
                whileHover={{ scale: 1.1 }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                {progress.problemsThisWeek[index] || 0}
              </motion.div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );

  const renderTime = () => (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center">
        <Clock className="w-16 h-16 mx-auto text-green-500 mb-4" />
        <h3 className="text-2xl font-bold mb-2">Time Investment</h3>
        <p className="text-gray-600 dark:text-gray-400">
          Every minute of learning builds expertise
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        <Card className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-3xl font-bold text-green-600">{progress.timeSpentToday}</div>
              <div className="text-sm text-green-700 dark:text-green-300">Minutes Today</div>
            </div>
            <Calendar className="w-12 h-12 text-green-500" />
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold">{progress.timeSpentTotal}h</div>
              <div className="text-sm text-gray-500">Total Time</div>
            </div>
            <div className="text-right">
              <div className="text-lg font-semibold">{progress.averageTime.toFixed(1)} min</div>
              <div className="text-sm text-gray-500">Avg per problem</div>
            </div>
          </div>
        </Card>
      </div>
    </motion.div>
  );

  const renderAchievements = () => (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center">
        <Award className="w-16 h-16 mx-auto text-yellow-500 mb-4" />
        <h3 className="text-2xl font-bold mb-2">Achievements</h3>
        <p className="text-gray-600 dark:text-gray-400">
          Celebrate your learning milestones
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {progress.achievements.map((achievement, index) => (
          <motion.div
            key={achievement.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className={cn(
              "p-4 border-l-4",
              achievement.rarity === 'legendary' && "border-l-yellow-500 bg-yellow-50 dark:bg-yellow-950",
              achievement.rarity === 'epic' && "border-l-purple-500 bg-purple-50 dark:bg-purple-950",
              achievement.rarity === 'rare' && "border-l-blue-500 bg-blue-50 dark:bg-blue-950",
              achievement.rarity === 'common' && "border-l-gray-500 bg-gray-50 dark:bg-gray-950"
            )}>
              <div className="flex items-center gap-4">
                <div className="text-2xl">{achievement.icon}</div>
                <div className="flex-1">
                  <h4 className="font-semibold">{achievement.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{achievement.description}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Unlocked {new Date(achievement.unlockedAt).toLocaleDateString()}
                  </p>
                </div>
                <div className={cn(
                  "px-2 py-1 rounded-full text-xs font-medium",
                  achievement.rarity === 'legendary' && "bg-yellow-200 text-yellow-800",
                  achievement.rarity === 'epic' && "bg-purple-200 text-purple-800",
                  achievement.rarity === 'rare' && "bg-blue-200 text-blue-800",
                  achievement.rarity === 'common' && "bg-gray-200 text-gray-800"
                )}>
                  {achievement.rarity}
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );

  return (
    <Card className="p-6 max-w-2xl mx-auto">
      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2 mb-6 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
        {[
          { key: 'overview', label: 'Overview', icon: TrendingUp },
          { key: 'streak', label: 'Streak', icon: Flame },
          { key: 'time', label: 'Time', icon: Clock },
          { key: 'achievements', label: 'Awards', icon: Award }
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setSelectedMetric(key as any)}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-200 text-sm font-medium",
              selectedMetric === key 
                ? "bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm"
                : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        {selectedMetric === 'overview' && renderOverview()}
        {selectedMetric === 'streak' && renderStreak()}
        {selectedMetric === 'time' && renderTime()}
        {selectedMetric === 'achievements' && renderAchievements()}
      </AnimatePresence>

      {/* Save to Google CTA */}
      {!isSignedIn && onSaveToGoogle && (
        <motion.div
          className="mt-8 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 rounded-lg border border-blue-200 dark:border-blue-800"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="text-center">
            <p className="text-sm text-blue-700 dark:text-blue-300 mb-3">
              💡 Save this progress to Google Drive to never lose your achievements!
            </p>
            <button
              onClick={onSaveToGoogle}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-2 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 text-sm font-medium"
            >
              Connect Google Drive
            </button>
          </div>
        </motion.div>
      )}
    </Card>
  );
};