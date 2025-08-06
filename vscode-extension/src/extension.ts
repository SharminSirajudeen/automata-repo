import * as vscode from 'vscode';
import { AutomataPreviewProvider } from './previewProvider';
import { AutomataLanguageClient } from './languageClient';
import { AutomataSimulator } from './simulator';
import { AutomataConverter } from './converter';
import { AutomataMinimizer } from './minimizer';
import { AutomataExporter } from './exporter';
import { AutomataValidator } from './simpleValidator';

let automataLanguageClient: AutomataLanguageClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('Automata Development extension is now active!');

    // Initialize preview provider
    const previewProvider = new AutomataPreviewProvider(context.extensionUri);
    
    // Register preview provider
    context.subscriptions.push(
        vscode.window.registerWebviewPanelSerializer(AutomataPreviewProvider.viewType, previewProvider)
    );

    // Initialize language client
    automataLanguageClient = new AutomataLanguageClient(context);
    automataLanguageClient.start();

    // Initialize other components
    const simulator = new AutomataSimulator();
    const converter = new AutomataConverter();
    const minimizer = new AutomataMinimizer();
    const exporter = new AutomataExporter();
    const validator = new AutomataValidator();

    // Register commands
    registerCommands(context, previewProvider, simulator, converter, minimizer, exporter, validator);

    // Register providers
    registerProviders(context, validator);

    // Set up document change listeners
    setupDocumentListeners(context, previewProvider, validator);
}

export function deactivate(): Thenable<void> | undefined {
    if (!automataLanguageClient) {
        return undefined;
    }
    return automataLanguageClient.stop();
}

function registerCommands(
    context: vscode.ExtensionContext,
    previewProvider: AutomataPreviewProvider,
    simulator: AutomataSimulator,
    converter: AutomataConverter,
    minimizer: AutomataMinimizer,
    exporter: AutomataExporter,
    validator: AutomataValidator
) {
    // Show preview command
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

    // Simulate command
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.simulate', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !isAutomataFile(editor.document)) {
                vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
                return;
            }
            await simulator.simulate(editor.document);
        })
    );

    // Test input command
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.testInput', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !isAutomataFile(editor.document)) {
                vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
                return;
            }
            
            const input = await vscode.window.showInputBox({
                prompt: 'Enter input string to test',
                placeHolder: 'Input string (e.g., "101", "abc")'
            });
            
            if (input !== undefined) {
                await simulator.testInput(editor.document, input);
            }
        })
    );

    // Minimize DFA command
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.minimize', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'automata-dfa') {
                vscode.window.showErrorMessage('Please open a DFA file first (.dfa)');
                return;
            }
            await minimizer.minimize(editor.document);
        })
    );

    // Convert format command
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.convert', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !isAutomataFile(editor.document)) {
                vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
                return;
            }
            await converter.convert(editor.document);
        })
    );

    // Validate command
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.validate', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !isAutomataFile(editor.document)) {
                vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
                return;
            }
            await validator.validateDocument(editor.document);
        })
    );

    // Export command
    context.subscriptions.push(
        vscode.commands.registerCommand('automata.export', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !isAutomataFile(editor.document)) {
                vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
                return;
            }
            await exporter.export(editor.document);
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

function registerProviders(context: vscode.ExtensionContext, validator: AutomataValidator) {
    // Register completion item provider
    const completionProvider = vscode.languages.registerCompletionItemProvider(
        [
            { scheme: 'file', language: 'automata-dfa' },
            { scheme: 'file', language: 'automata-nfa' },
            { scheme: 'file', language: 'automata-tm' }
        ],
        {
            provideCompletionItems(document: vscode.TextDocument, position: vscode.Position) {
                return getCompletionItems(document, position);
            }
        },
        ':', '{', '"'
    );

    // Register hover provider
    const hoverProvider = vscode.languages.registerHoverProvider(
        [
            { scheme: 'file', language: 'automata-dfa' },
            { scheme: 'file', language: 'automata-nfa' },
            { scheme: 'file', language: 'automata-tm' }
        ],
        {
            provideHover(document, position) {
                return getHoverInformation(document, position);
            }
        }
    );

    // Register diagnostic provider
    const diagnosticCollection = vscode.languages.createDiagnosticCollection('automata');
    context.subscriptions.push(diagnosticCollection);

    context.subscriptions.push(completionProvider, hoverProvider);
}

function setupDocumentListeners(
    context: vscode.ExtensionContext,
    previewProvider: AutomataPreviewProvider,
    validator: AutomataValidator
) {
    // Listen for document changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(async (event) => {
            if (isAutomataFile(event.document)) {
                const config = vscode.workspace.getConfiguration('automata');
                
                // Auto-refresh preview if enabled
                if (config.get('preview.autoRefresh', true)) {
                    previewProvider.updatePreview(event.document.uri);
                }
                
                // Real-time validation if enabled
                if (config.get('validation.enableRealtime', true)) {
                    await validator.validateDocument(event.document);
                }
            }
        })
    );

    // Listen for document saves
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(async (document) => {
            if (isAutomataFile(document)) {
                await validator.validateDocument(document);
                previewProvider.updatePreview(document.uri);
            }
        })
    );
}

function isAutomataFile(document: vscode.TextDocument): boolean {
    return ['automata-dfa', 'automata-nfa', 'automata-tm'].includes(document.languageId);
}

function getCompletionItems(document: vscode.TextDocument, position: vscode.Position): vscode.CompletionItem[] {
    const items: vscode.CompletionItem[] = [];
    const line = document.lineAt(position).text;
    const beforeCursor = line.substring(0, position.character);

    // Common keywords for all automata types
    const commonKeywords = [
        { label: 'states', detail: 'Define the set of states' },
        { label: 'alphabet', detail: 'Define the input alphabet' },
        { label: 'start_state', detail: 'Define the initial state' },
        { label: 'accept_states', detail: 'Define accepting/final states' },
        { label: 'transitions', detail: 'Define state transitions' }
    ];

    // Language-specific keywords
    if (document.languageId === 'automata-tm') {
        commonKeywords.push(
            { label: 'tape_alphabet', detail: 'Define the tape alphabet including blank symbol' },
            { label: 'blank_symbol', detail: 'Define the blank tape symbol' }
        );
    }

    // Add completions based on context
    if (beforeCursor.includes('"')) {
        // Inside a string, suggest state names or symbols
        const stateNames = ['q0', 'q1', 'q2', 'qaccept', 'qreject'];
        const symbols = ['0', '1', 'a', 'b', 'epsilon', '_'];
        
        stateNames.forEach(state => {
            const item = new vscode.CompletionItem(state, vscode.CompletionItemKind.Value);
            item.detail = 'State name';
            items.push(item);
        });

        symbols.forEach(symbol => {
            const item = new vscode.CompletionItem(symbol, vscode.CompletionItemKind.Value);
            item.detail = 'Symbol';
            items.push(item);
        });
    } else {
        // Add keyword completions
        commonKeywords.forEach(keyword => {
            const item = new vscode.CompletionItem(keyword.label, vscode.CompletionItemKind.Keyword);
            item.detail = keyword.detail;
            item.insertText = `"${keyword.label}": `;
            items.push(item);
        });
    }

    return items;
}

function getHoverInformation(document: vscode.TextDocument, position: vscode.Position): vscode.Hover | undefined {
    const range = document.getWordRangeAtPosition(position);
    if (!range) return undefined;

    const word = document.getText(range);
    
    const hoverInfo: { [key: string]: string } = {
        'states': 'The finite set of states in the automaton. Each state represents a configuration of the computation.',
        'alphabet': 'The finite set of input symbols that the automaton can read.',
        'start_state': 'The initial state where computation begins.',
        'accept_states': 'The set of final/accepting states. If computation ends in one of these states, the input is accepted.',
        'transitions': 'The transition function that defines how the automaton moves from one state to another based on input.',
        'tape_alphabet': 'The complete set of symbols that can appear on the tape, including input symbols and the blank symbol.',
        'blank_symbol': 'The symbol used to represent empty cells on the tape.',
        'epsilon': 'Represents an empty string or ε-transition (transition without consuming input).'
    };

    if (hoverInfo[word]) {
        return new vscode.Hover(hoverInfo[word]);
    }

    return undefined;
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
        const uri = vscode.Uri.file(`${fileName}.${type}`);
        const document = await vscode.workspace.openTextDocument(uri);
        const editor = await vscode.window.showTextDocument(document);
        
        await editor.edit(editBuilder => {
            editBuilder.insert(new vscode.Position(0, 0), templates[type]);
        });
    }
}