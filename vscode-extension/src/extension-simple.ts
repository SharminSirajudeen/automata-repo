import * as vscode from 'vscode';
import { AutomataPreviewProvider } from './previewProvider-simple';

export function activate(context: vscode.ExtensionContext) {
    console.log('Automata Development extension is now active!');

    // Initialize preview provider
    const previewProvider = new AutomataPreviewProvider(context.extensionUri);
    
    // Register preview provider
    context.subscriptions.push(
        vscode.window.registerWebviewPanelSerializer(AutomataPreviewProvider.viewType, previewProvider)
    );

    // Register show preview command
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.showPreview', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !isAutomataFile(editor.document)) {
                vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
                return;
            }
            previewProvider.showPreview(editor.document.uri, editor.viewColumn);
        })
    );

    // Register other basic commands
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.simulate', () => {
            vscode.window.showInformationMessage('Simulation feature coming soon!');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('automata.minimize', () => {
            vscode.window.showInformationMessage('Minimization feature coming soon!');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('automata.convert', () => {
            vscode.window.showInformationMessage('Conversion feature coming soon!');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('automata.validate', () => {
            vscode.window.showInformationMessage('Validation feature coming soon!');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('automata.export', () => {
            vscode.window.showInformationMessage('Export feature coming soon!');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('automata.testInput', () => {
            vscode.window.showInformationMessage('Test input feature coming soon!');
        })
    );

    // Create new automata commands
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.createDFA', () => createNewAutomata('dfa'))
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.createNFA', () => createNewAutomata('nfa'))
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.createTM', () => createNewAutomata('tm'))
    );
}

export function deactivate(): Thenable<void> | undefined {
    return undefined;
}

function isAutomataFile(document: vscode.TextDocument): boolean {
    return ['automata-dfa', 'automata-nfa', 'automata-tm'].includes(document.languageId) ||
           document.fileName.endsWith('.dfa') || 
           document.fileName.endsWith('.nfa') || 
           document.fileName.endsWith('.tm');
}

async function createNewAutomata(type: 'dfa' | 'nfa' | 'tm') {
    const templates = {
        dfa: `{
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start_state": "q0",
  "accept_states": ["q2"],
  "transitions": {
    "q0": {
      "0": "q1",
      "1": "q0"
    },
    "q1": {
      "0": "q2",
      "1": "q0"
    },
    "q2": {
      "0": "q2",
      "1": "q2"
    }
  }
}`,
        nfa: `{
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start_state": "q0",
  "accept_states": ["q2"],
  "transitions": {
    "q0": {
      "0": ["q0", "q1"],
      "1": ["q0"],
      "epsilon": ["q1"]
    },
    "q1": {
      "1": ["q2"]
    },
    "q2": {}
  }
}`,
        tm: `{
  "states": ["q0", "q1", "q2", "qaccept", "qreject"],
  "alphabet": ["0", "1"],
  "tape_alphabet": ["0", "1", "_"],
  "blank_symbol": "_",
  "start_state": "q0",
  "accept_states": ["qaccept"],
  "reject_states": ["qreject"],
  "transitions": {
    "q0": {
      "0": {
        "state": "q1",
        "write": "X",
        "move": "R"
      },
      "1": {
        "state": "qreject",
        "write": "1",
        "move": "R"
      },
      "_": {
        "state": "qaccept",
        "write": "_",
        "move": "R"
      }
    },
    "q1": {
      "0": {
        "state": "q1",
        "write": "0",
        "move": "R"
      },
      "1": {
        "state": "q2",
        "write": "Y",
        "move": "L"
      },
      "_": {
        "state": "qreject",
        "write": "_",
        "move": "R"
      }
    },
    "q2": {
      "0": {
        "state": "q2",
        "write": "0",
        "move": "L"
      },
      "X": {
        "state": "q0",
        "write": "X",
        "move": "R"
      },
      "Y": {
        "state": "q2",
        "write": "Y",
        "move": "L"
      }
    }
  }
}`
    };

    const fileName = await vscode.window.showInputBox({
        prompt: `Enter name for new ${type.toUpperCase()} file`,
        placeHolder: `my_${type}`,
        validateInput: (input) => {
            if (!input || input.trim() === '') {
                return 'File name cannot be empty';
            }
            if (!/^[a-zA-Z0-9_-]+$/.test(input)) {
                return 'File name can only contain letters, numbers, underscores, and hyphens';
            }
            return null;
        }
    });

    if (fileName) {
        const document = await vscode.workspace.openTextDocument({
            content: templates[type],
            language: 'json'
        });
        await vscode.window.showTextDocument(document);
    }
}