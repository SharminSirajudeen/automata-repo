"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const previewProvider_1 = require("./previewProvider");
const languageClient_1 = require("./languageClient");
const simulator_1 = require("./simulator");
const converter_1 = require("./converter");
const minimizer_1 = require("./minimizer");
const exporter_1 = require("./exporter");
const simpleValidator_1 = require("./simpleValidator");
let automataLanguageClient;
function activate(context) {
    console.log('Automata Development extension is now active!');
    // Initialize preview provider
    const previewProvider = new previewProvider_1.AutomataPreviewProvider(context.extensionUri);
    // Register preview provider
    context.subscriptions.push(vscode.window.registerWebviewPanelSerializer(previewProvider_1.AutomataPreviewProvider.viewType, previewProvider));
    // Initialize language client
    automataLanguageClient = new languageClient_1.AutomataLanguageClient(context);
    automataLanguageClient.start();
    // Initialize other components
    const simulator = new simulator_1.AutomataSimulator();
    const converter = new converter_1.AutomataConverter();
    const minimizer = new minimizer_1.AutomataMinimizer();
    const exporter = new exporter_1.AutomataExporter();
    const validator = new simpleValidator_1.AutomataValidator();
    // Register commands
    registerCommands(context, previewProvider, simulator, converter, minimizer, exporter, validator);
    // Register providers
    registerProviders(context, validator);
    // Set up document change listeners
    setupDocumentListeners(context, previewProvider, validator);
}
exports.activate = activate;
function deactivate() {
    if (!automataLanguageClient) {
        return undefined;
    }
    return automataLanguageClient.stop();
}
exports.deactivate = deactivate;
function registerCommands(context, previewProvider, simulator, converter, minimizer, exporter, validator) {
    // Show preview command
    context.subscriptions.push(vscode.commands.registerCommand('automata.showPreview', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !isAutomataFile(editor.document)) {
            vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
            return;
        }
        previewProvider.showPreview(editor.document.uri, editor.viewColumn);
    }));
    // Simulate command
    context.subscriptions.push(vscode.commands.registerCommand('automata.simulate', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !isAutomataFile(editor.document)) {
            vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
            return;
        }
        await simulator.simulate(editor.document);
    }));
    // Test input command
    context.subscriptions.push(vscode.commands.registerCommand('automata.testInput', async () => {
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
    }));
    // Minimize DFA command
    context.subscriptions.push(vscode.commands.registerCommand('automata.minimize', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'automata-dfa') {
            vscode.window.showErrorMessage('Please open a DFA file first (.dfa)');
            return;
        }
        await minimizer.minimize(editor.document);
    }));
    // Convert format command
    context.subscriptions.push(vscode.commands.registerCommand('automata.convert', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !isAutomataFile(editor.document)) {
            vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
            return;
        }
        await converter.convert(editor.document);
    }));
    // Validate command
    context.subscriptions.push(vscode.commands.registerCommand('automata.validate', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !isAutomataFile(editor.document)) {
            vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
            return;
        }
        await validator.validateDocument(editor.document);
    }));
    // Export command
    context.subscriptions.push(vscode.commands.registerCommand('automata.export', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !isAutomataFile(editor.document)) {
            vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
            return;
        }
        await exporter.export(editor.document);
    }));
    // Create new automata commands
    context.subscriptions.push(vscode.commands.registerCommand('automata.createDFA', () => createNewAutomata('dfa')));
    context.subscriptions.push(vscode.commands.registerCommand('automata.createNFA', () => createNewAutomata('nfa')));
    context.subscriptions.push(vscode.commands.registerCommand('automata.createTM', () => createNewAutomata('tm')));
}
function registerProviders(context, validator) {
    // Register completion item provider
    const completionProvider = vscode.languages.registerCompletionItemProvider([
        { scheme: 'file', language: 'automata-dfa' },
        { scheme: 'file', language: 'automata-nfa' },
        { scheme: 'file', language: 'automata-tm' }
    ], {
        provideCompletionItems(document, position) {
            return getCompletionItems(document, position);
        }
    }, ':', '{', '"');
    // Register hover provider
    const hoverProvider = vscode.languages.registerHoverProvider([
        { scheme: 'file', language: 'automata-dfa' },
        { scheme: 'file', language: 'automata-nfa' },
        { scheme: 'file', language: 'automata-tm' }
    ], {
        provideHover(document, position) {
            return getHoverInformation(document, position);
        }
    });
    // Register diagnostic provider
    const diagnosticCollection = vscode.languages.createDiagnosticCollection('automata');
    context.subscriptions.push(diagnosticCollection);
    context.subscriptions.push(completionProvider, hoverProvider);
}
function setupDocumentListeners(context, previewProvider, validator) {
    // Listen for document changes
    context.subscriptions.push(vscode.workspace.onDidChangeTextDocument(async (event) => {
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
    }));
    // Listen for document saves
    context.subscriptions.push(vscode.workspace.onDidSaveTextDocument(async (document) => {
        if (isAutomataFile(document)) {
            await validator.validateDocument(document);
            previewProvider.updatePreview(document.uri);
        }
    }));
}
function isAutomataFile(document) {
    return ['automata-dfa', 'automata-nfa', 'automata-tm'].includes(document.languageId);
}
function getCompletionItems(document, position) {
    const items = [];
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
        commonKeywords.push({ label: 'tape_alphabet', detail: 'Define the tape alphabet including blank symbol' }, { label: 'blank_symbol', detail: 'Define the blank tape symbol' });
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
    }
    else {
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
function getHoverInformation(document, position) {
    const range = document.getWordRangeAtPosition(position);
    if (!range)
        return undefined;
    const word = document.getText(range);
    const hoverInfo = {
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
async function createNewAutomata(type) {
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
//# sourceMappingURL=extension.js.map