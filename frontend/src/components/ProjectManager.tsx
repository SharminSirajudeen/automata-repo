import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Separator } from './ui/separator';
import { FolderOpen, Save, Download, Upload, Trash2, Plus, FileText, Clock } from 'lucide-react';
import { ExtendedAutomaton, AutomataType, Problem } from '../types/automata';

interface Project {
  id: string;
  name: string;
  type: AutomataType;
  automaton: ExtendedAutomaton;
  problem?: Problem;
  created: Date;
  modified: Date;
}

interface ProjectManagerProps {
  currentAutomaton?: ExtendedAutomaton;
  currentProblem?: Problem;
  onLoadProject: (project: Project) => void;
  onSaveProject: (name: string) => void;
}

export const ProjectManager: React.FC<ProjectManagerProps> = ({
  currentAutomaton,
  currentProblem,
  onLoadProject,
  onSaveProject
}) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [newProjectName, setNewProjectName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateProject = () => {
    if (!newProjectName.trim() || !currentAutomaton) return;

    const newProject: Project = {
      id: Date.now().toString(),
      name: newProjectName.trim(),
      type: currentAutomaton.type as AutomataType,
      automaton: currentAutomaton,
      problem: currentProblem,
      created: new Date(),
      modified: new Date()
    };

    setProjects(prev => [newProject, ...prev]);
    setNewProjectName('');
    setIsCreating(false);
    onSaveProject(newProject.name);
  };

  const handleDeleteProject = (projectId: string) => {
    setProjects(prev => prev.filter(p => p.id !== projectId));
  };

  const handleExportProject = (project: Project) => {
    const exportData = {
      name: project.name,
      type: project.type,
      automaton: project.automaton,
      problem: project.problem,
      metadata: {
        created: project.created,
        modified: project.modified,
        version: '1.0'
      }
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.name}.jflap.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImportProject = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importData = JSON.parse(e.target?.result as string);
        const importedProject: Project = {
          id: Date.now().toString(),
          name: importData.name || 'Imported Project',
          type: importData.type,
          automaton: importData.automaton,
          problem: importData.problem,
          created: new Date(importData.metadata?.created || Date.now()),
          modified: new Date()
        };

        setProjects(prev => [importedProject, ...prev]);
      } catch (error) {
        console.error('Failed to import project:', error);
      }
    };
    reader.readAsText(file);
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

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <FolderOpen className="w-5 h-5 text-blue-600" />
          Project Manager
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button
            onClick={() => setIsCreating(true)}
            disabled={!currentAutomaton}
            className="flex-1"
            size="sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Project
          </Button>
          
          <label className="cursor-pointer">
            <Button variant="outline" size="sm" asChild>
              <span>
                <Upload className="w-4 h-4 mr-2" />
                Import
              </span>
            </Button>
            <input
              type="file"
              accept=".json,.jflap"
              onChange={handleImportProject}
              className="hidden"
            />
          </label>
        </div>

        {isCreating && (
          <div className="space-y-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <Input
              placeholder="Project name..."
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleCreateProject()}
            />
            <div className="flex gap-2">
              <Button onClick={handleCreateProject} size="sm" className="flex-1">
                <Save className="w-4 h-4 mr-2" />
                Save Project
              </Button>
              <Button 
                onClick={() => {
                  setIsCreating(false);
                  setNewProjectName('');
                }}
                variant="outline" 
                size="sm"
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        <Separator />

        <div className="space-y-2 max-h-96 overflow-y-auto">
          {projects.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No projects yet</p>
              <p className="text-xs">Create your first project to get started</p>
            </div>
          ) : (
            projects.map(project => (
              <div key={project.id} className="p-3 border rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h3 className="font-medium text-sm">{project.name}</h3>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="secondary" className="text-xs">
                        {getTypeDisplayName(project.type)}
                      </Badge>
                      <div className="flex items-center gap-1 text-xs text-gray-500">
                        <Clock className="w-3 h-3" />
                        {formatDate(project.modified)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex gap-1">
                    <Button
                      onClick={() => onLoadProject(project)}
                      size="sm"
                      variant="outline"
                      className="h-7 px-2"
                    >
                      <FolderOpen className="w-3 h-3" />
                    </Button>
                    <Button
                      onClick={() => handleExportProject(project)}
                      size="sm"
                      variant="outline"
                      className="h-7 px-2"
                    >
                      <Download className="w-3 h-3" />
                    </Button>
                    <Button
                      onClick={() => handleDeleteProject(project.id)}
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
                
                {project.problem && (
                  <p className="text-xs text-gray-600 truncate">
                    Problem: {project.problem.title}
                  </p>
                )}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};
