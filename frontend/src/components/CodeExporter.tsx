import React, { useState } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { Code2, Copy, Download, FileText } from 'lucide-react';
import { ExtendedAutomaton, AutomataType, CodeExportOptions, ExportResult } from '../types/automata';

interface CodeExporterProps {
  automaton: ExtendedAutomaton;
  automatonType: AutomataType;
  onCodeGenerated?: (code: string, language: string) => void;
}

export const CodeExporter: React.FC<CodeExporterProps> = ({
  automaton,
  automatonType,
  onCodeGenerated
}) => {
  const [exportOptions, setExportOptions] = useState<CodeExportOptions>({
    language: 'python',
    include_tests: true,
    include_visualization: false,
    format: 'class'
  });
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    if (!automaton) return;

    setIsExporting(true);
    try {
      const response = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          automaton,
          options: exportOptions
        })
      });

      if (response.ok) {
        const result: ExportResult = await response.json();
        setExportResult(result);
        onCodeGenerated?.(result.code, result.language);
      }
    } catch (error) {
      console.error('Export error:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const handleCopyCode = () => {
    if (exportResult) {
      navigator.clipboard.writeText(exportResult.code);
    }
  };

  const handleDownload = () => {
    if (!exportResult) return;

    const blob = new Blob([exportResult.code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = exportResult.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getLanguageIcon = (language: string) => {
    switch (language) {
      case 'python': return '🐍';
      case 'javascript': return '🟨';
      case 'java': return '☕';
      default: return '📄';
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
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Code2 className="w-5 h-5 text-purple-600" />
          Code Exporter
          <Badge variant="outline" className="ml-auto">
            {getTypeDisplayName(automatonType)}
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">Language</label>
            <Select
              value={exportOptions.language}
              onValueChange={(value) => setExportOptions(prev => ({ ...prev, language: value as any }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="python">
                  <div className="flex items-center gap-2">
                    <span>🐍</span>
                    Python
                  </div>
                </SelectItem>
                <SelectItem value="javascript">
                  <div className="flex items-center gap-2">
                    <span>🟨</span>
                    JavaScript
                  </div>
                </SelectItem>
                <SelectItem value="java">
                  <div className="flex items-center gap-2">
                    <span>☕</span>
                    Java
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">Format</label>
            <Select
              value={exportOptions.format}
              onValueChange={(value) => setExportOptions(prev => ({ ...prev, format: value as any }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="class">Class-based</SelectItem>
                <SelectItem value="function">Function-based</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Checkbox
              id="include_tests"
              checked={exportOptions.include_tests}
              onCheckedChange={(checked) => 
                setExportOptions(prev => ({ ...prev, include_tests: checked as boolean }))
              }
            />
            <label htmlFor="include_tests" className="text-sm font-medium">
              Include test cases
            </label>
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="include_visualization"
              checked={exportOptions.include_visualization}
              onCheckedChange={(checked) => 
                setExportOptions(prev => ({ ...prev, include_visualization: checked as boolean }))
              }
            />
            <label htmlFor="include_visualization" className="text-sm font-medium">
              Include visualization code
            </label>
          </div>
        </div>

        <Button 
          onClick={handleExport} 
          disabled={isExporting}
          className="w-full"
        >
          {isExporting ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              Generating...
            </>
          ) : (
            <>
              <Code2 className="w-4 h-4 mr-2" />
              Generate {exportOptions.language} Code
            </>
          )}
        </Button>

        {exportResult && (
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2">
                <span className="text-lg">{getLanguageIcon(exportResult.language)}</span>
                <div>
                  <p className="font-medium text-sm">{exportResult.filename}</p>
                  <p className="text-xs text-gray-600">
                    {exportResult.language.charAt(0).toUpperCase() + exportResult.language.slice(1)} Code
                  </p>
                </div>
              </div>
              
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={handleCopyCode}>
                  <Copy className="w-3 h-3 mr-1" />
                  Copy
                </Button>
                <Button size="sm" variant="outline" onClick={handleDownload}>
                  <Download className="w-3 h-3 mr-1" />
                  Download
                </Button>
              </div>
            </div>

            <div className="max-h-96 overflow-y-auto">
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
                <code>{exportResult.code}</code>
              </pre>
            </div>

            {exportResult.test_cases && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <FileText className="w-4 h-4 text-gray-600" />
                  <span className="text-sm font-medium text-gray-700">Test Cases</span>
                </div>
                <div className="max-h-48 overflow-y-auto">
                  <pre className="bg-gray-100 text-gray-800 p-3 rounded-lg text-sm overflow-x-auto">
                    <code>{exportResult.test_cases}</code>
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="text-xs text-gray-500 space-y-1">
          <p>💡 <strong>Tip:</strong> The generated code includes a complete simulator for your {getTypeDisplayName(automatonType)}.</p>
          <p>🔧 You can modify the code to add custom features or integrate with your projects.</p>
          <p>📚 Test cases help verify your automaton works correctly with various inputs.</p>
        </div>
      </CardContent>
    </Card>
  );
};
