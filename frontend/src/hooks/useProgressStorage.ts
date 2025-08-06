import { useState, useEffect } from 'react';
import { useToast } from '@/components/ui/use-toast';

// Simple progress tracking
interface Progress {
  completedProblems: string[];
  currentStreak: number;
  lastActive: string;
}

export const useProgressStorage = () => {
  const [progress, setProgress] = useState<Progress>({
    completedProblems: [],
    currentStreak: 0,
    lastActive: new Date().toISOString()
  });
  const { toast } = useToast();

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('automata-progress');
    if (saved) {
      setProgress(JSON.parse(saved));
    }
  }, []);

  // Save to localStorage
  const saveProgress = (newProgress: Progress) => {
    setProgress(newProgress);
    localStorage.setItem('automata-progress', JSON.stringify(newProgress));
  };

  // Mark problem as completed
  const completeProblem = (problemId: string) => {
    if (!progress.completedProblems.includes(problemId)) {
      const updated = {
        ...progress,
        completedProblems: [...progress.completedProblems, problemId],
        lastActive: new Date().toISOString()
      };
      saveProgress(updated);
      
      // Show completion message
      if (updated.completedProblems.length % 5 === 0) {
        toast({
          title: `${updated.completedProblems.length} Problems Completed! 🎉`,
          description: "Keep up the great work!"
        });
      }
    }
  };

  return {
    progress,
    completeProblem,
    isCompleted: (problemId: string) => progress.completedProblems.includes(problemId)
  };
};