import React, { useState } from 'react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Upload, FileText, Image, Sparkles } from 'lucide-react';

interface ProblemInputProps {
  onProblemSubmit: (problem: string, type: 'text' | 'image') => void;
  isProcessing?: boolean;
}

export const ProblemInput: React.FC<ProblemInputProps> = ({ 
  onProblemSubmit, 
  isProcessing = false 
}) => {
  const [problemText, setProblemText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<'text' | 'image'>('text');

  const handleTextSubmit = () => {
    if (problemText.trim()) {
      onProblemSubmit(problemText.trim(), 'text');
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
    }
  };

  const handleImageSubmit = () => {
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = e.target?.result as string;
        onProblemSubmit(base64, 'image');
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-blue-500" />
          AI-Powered Problem Solver
        </CardTitle>
        <CardDescription>
          Enter a Theory of Computation problem in natural language or upload an image from a textbook. 
          Our AI will analyze it and provide either a guided solution or complete answer.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2 mb-4">
          <Button
            variant={inputMode === 'text' ? 'default' : 'outline'}
            onClick={() => setInputMode('text')}
            className="flex items-center gap-2"
          >
            <FileText className="h-4 w-4" />
            Text Input
          </Button>
          <Button
            variant={inputMode === 'image' ? 'default' : 'outline'}
            onClick={() => setInputMode('image')}
            className="flex items-center gap-2"
          >
            <Image className="h-4 w-4" />
            Image Upload
          </Button>
        </div>

        {inputMode === 'text' ? (
          <div className="space-y-4">
            <Textarea
              placeholder="Enter your Theory of Computation problem here...

Examples:
• Construct a DFA that accepts strings ending in 'ab'
• Design a PDA for balanced parentheses
• Create a Turing machine that computes n+1
• Build a CFG for the language {a^n b^n | n ≥ 0}
• Prove that the language L = {ww | w ∈ {0,1}*} is not regular using the pumping lemma"
              value={problemText}
              onChange={(e) => setProblemText(e.target.value)}
              className="min-h-[200px] resize-none"
              disabled={isProcessing}
            />
            <Button
              onClick={handleTextSubmit}
              disabled={!problemText.trim() || isProcessing}
              className="w-full"
            >
              {isProcessing ? 'Analyzing Problem...' : 'Solve Problem'}
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
                id="image-upload"
                disabled={isProcessing}
              />
              <label htmlFor="image-upload" className="cursor-pointer">
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-700">
                  Upload Problem Image
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Click to select an image from your textbook or homework
                </p>
              </label>
              {selectedFile && (
                <div className="mt-4 p-4 bg-green-50 rounded-lg">
                  <p className="text-green-700 font-medium">
                    Selected: {selectedFile.name}
                  </p>
                  <p className="text-green-600 text-sm">
                    Size: {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              )}
            </div>
            <Button
              onClick={handleImageSubmit}
              disabled={!selectedFile || isProcessing}
              className="w-full"
            >
              {isProcessing ? 'Processing Image...' : 'Analyze Image'}
            </Button>
          </div>
        )}

        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">What our AI can do:</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Automatically detect if your problem is Theory of Computation related</li>
            <li>• Generate complete solutions with step-by-step explanations</li>
            <li>• Provide guided approaches that walk you through the solution</li>
            <li>• Support all automata types: DFA, NFA, PDA, TM, CFG, Regex, Pumping Lemma</li>
            <li>• Extract problems from textbook images using llava:34b vision model</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};
