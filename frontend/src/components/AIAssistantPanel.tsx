import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Send, Bot, User, Loader2, Download, Play, Eye } from 'lucide-react';
import { AutomataType, ExtendedAutomaton } from '../types/automata';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  automaton?: ExtendedAutomaton;
  code?: string;
  language?: string;
}

interface AIAssistantPanelProps {
  automatonType: AutomataType;
  currentAutomaton?: ExtendedAutomaton;
  onAutomatonGenerated?: (automaton: ExtendedAutomaton) => void;
  onCodeExported?: (code: string, language: string) => void;
}

export const AIAssistantPanel: React.FC<AIAssistantPanelProps> = ({
  automatonType,
  currentAutomaton,
  onAutomatonGenerated,
  onCodeExported
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: `Hello! I'm your AI assistant for ${automatonType.toUpperCase()} problems. I can help you:
      
• Generate automata from natural language descriptions
• Explain how automata work step by step  
• Export your automata to Python, JavaScript, or Java code
• Provide hints and guidance for construction
• Simulate and trace execution paths

What would you like to work on today?`,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await processUserMessage(inputValue);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.content,
        timestamp: new Date(),
        automaton: response.automaton,
        code: response.code,
        language: response.language
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (response.automaton && onAutomatonGenerated) {
        onAutomatonGenerated(response.automaton);
      }

      if (response.code && response.language && onCodeExported) {
        onCodeExported(response.code, response.language);
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const processUserMessage = async (message: string): Promise<{
    content: string;
    automaton?: ExtendedAutomaton;
    code?: string;
    language?: string;
  }> => {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('generate') || lowerMessage.includes('create') || lowerMessage.includes('build')) {
      try {
        const response = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task: message,
            type: automatonType
          })
        });

        if (response.ok) {
          const data = await response.json();
          return {
            content: `I've generated a ${automatonType.toUpperCase()} for your description. Here's what I created:

**Formal Definition:**
${data.generated_automaton.formal_definition || 'Generated automaton structure'}

**Explanation:**
${data.generated_automaton.explanation || 'This automaton recognizes the specified language.'}

The automaton has been applied to your canvas. You can modify it as needed!`,
            automaton: data.generated_automaton.automaton
          };
        }
      } catch (error) {
        console.error('Generation error:', error);
      }
    }

    if (lowerMessage.includes('export') || lowerMessage.includes('code')) {
      const language = lowerMessage.includes('python') ? 'python' :
                      lowerMessage.includes('javascript') || lowerMessage.includes('js') ? 'javascript' :
                      lowerMessage.includes('java') ? 'java' : 'python';

      try {
        const response = await fetch('/api/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            automaton: currentAutomaton,
            options: {
              language,
              include_tests: true,
              include_visualization: false,
              format: 'class'
            }
          })
        });

        if (response.ok) {
          const data = await response.json();
          return {
            content: `I've exported your ${automatonType.toUpperCase()} to ${language}! Here's the generated code:

\`\`\`${language}
${data.code}
\`\`\`

${data.test_cases ? `**Test Cases:**
\`\`\`${language}
${data.test_cases}
\`\`\`` : ''}

You can copy this code and run it in your ${language} environment.`,
            code: data.code,
            language
          };
        }
      } catch (error) {
        console.error('Export error:', error);
      }
    }

    if (lowerMessage.includes('explain') || lowerMessage.includes('how')) {
      try {
        const response = await fetch('/api/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            automaton: currentAutomaton,
            task: message
          })
        });

        if (response.ok) {
          const data = await response.json();
          return {
            content: `Here's how your ${automatonType.toUpperCase()} works:

**Overview:**
${data.explanation.overview || 'This automaton processes input strings according to its transition rules.'}

**State-by-State Explanation:**
${data.explanation.state_explanation || 'Each state has a specific role in recognizing the language.'}

**Example Walkthrough:**
${data.explanation.example_walkthrough || 'Let me trace through an example string...'}

Would you like me to simulate a specific input string?`
          };
        }
      } catch (error) {
        console.error('Explanation error:', error);
      }
    }

    if (lowerMessage.includes('simulate') || lowerMessage.includes('trace') || lowerMessage.includes('run')) {
      const inputString = extractInputString(message);
      if (inputString && currentAutomaton) {
        try {
          const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              automaton: currentAutomaton,
              input_string: inputString,
              type: automatonType
            })
          });

          if (response.ok) {
            const data = await response.json();
            return {
              content: `Simulation Results for "${inputString}":

**Result:** ${data.accepted ? '✅ ACCEPTED' : '❌ REJECTED'}

**Execution Path:**
${data.execution_path.map((step: string, i: number) => `${i + 1}. ${step}`).join('\n')}

${data.error_message ? `**Error:** ${data.error_message}` : ''}

The automaton ${data.accepted ? 'successfully accepted' : 'rejected'} the input string.`
            };
          }
        } catch (error) {
          console.error('Simulation error:', error);
        }
      }
    }

    return {
      content: `I understand you're asking about "${message}". Here are some things I can help you with:

🤖 **Generate Automata:** "Create a DFA that accepts strings ending in 'ab'"
📝 **Explain Concepts:** "How does this automaton work?"
💻 **Export Code:** "Export this to Python code"
🔍 **Simulate Input:** "Simulate the string 'abab'"
💡 **Get Hints:** "Give me a hint for this problem"

What would you like to explore?`
    };
  };

  const extractInputString = (message: string): string | null => {
    const patterns = [
      /"([^"]+)"/,
      /'([^']+)'/,
      /string\s+([a-zA-Z0-9]+)/i,
      /input\s+([a-zA-Z0-9]+)/i,
      /simulate\s+([a-zA-Z0-9]+)/i
    ];

    for (const pattern of patterns) {
      const match = message.match(pattern);
      if (match) return match[1];
    }
    return null;
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getTypeDisplayName = (type: AutomataType) => {
    const typeNames: { [key in AutomataType]: string } = {
      'dfa': 'DFA',
      'nfa': 'NFA',
      'enfa': 'ε-NFA',
      'pda': 'PDA',
      'cfg': 'CFG',
      'tm': 'Turing Machine',
      'regex': 'Regular Expression',
      'pumping': 'Pumping Lemma'
    };
    return typeNames[type];
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-blue-600" />
          AI Assistant
          <Badge variant="outline" className="ml-auto">
            {getTypeDisplayName(automatonType)}
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col p-0">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex gap-3 max-w-[80%] ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.type === 'user' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-600'
                }`}>
                  {message.type === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>
                
                <div className={`rounded-lg p-3 ${
                  message.type === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}>
                  <div className="whitespace-pre-wrap text-sm">{message.content}</div>
                  
                  {message.automaton && (
                    <div className="mt-2 pt-2 border-t border-gray-300">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => onAutomatonGenerated?.(message.automaton!)}
                        className="text-xs"
                      >
                        <Play className="w-3 h-3 mr-1" />
                        Apply to Canvas
                      </Button>
                    </div>
                  )}
                  
                  {message.code && (
                    <div className="mt-2 pt-2 border-t border-gray-300 flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => navigator.clipboard.writeText(message.code!)}
                        className="text-xs"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        Copy Code
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => onCodeExported?.(message.code!, message.language!)}
                        className="text-xs"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        View Full
                      </Button>
                    </div>
                  )}
                  
                  <div className="text-xs opacity-70 mt-1">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-gray-200 text-gray-600 flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-gray-100 text-gray-900 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        <div className="border-t p-4">
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about automata..."
              disabled={isLoading}
              className="flex-1"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              size="sm"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
          
          <div className="flex flex-wrap gap-1 mt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInputValue('Generate a DFA that accepts strings ending in "01"')}
              className="text-xs"
            >
              Generate Example
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInputValue('Explain how this automaton works')}
              className="text-xs"
            >
              Explain Current
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInputValue('Export to Python code')}
              className="text-xs"
            >
              Export Code
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
