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
const previewProvider_simple_1 = require("./previewProvider-simple");
function activate(context) {
    console.log('Automata Development extension is now active!');
    // Initialize preview provider
    const previewProvider = new previewProvider_simple_1.AutomataPreviewProvider(context.extensionUri);
    // Register preview provider
    context.subscriptions.push(vscode.window.registerWebviewPanelSerializer(previewProvider_simple_1.AutomataPreviewProvider.viewType, previewProvider));
    // Register show preview command
    context.subscriptions.push(vscode.commands.registerCommand('automata.showPreview', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !isAutomataFile(editor.document)) {
            vscode.window.showErrorMessage('Please open an automata file first (.dfa, .nfa, or .tm)');
            return;
        }
        previewProvider.showPreview(editor.document.uri, editor.viewColumn);
    }));
    // Register other basic commands
    context.subscriptions.push(vscode.commands.registerCommand('automata.simulate', () => {
        vscode.window.showInformationMessage('Simulation feature coming soon!');
    }));
    context.subscriptions.push(vscode.commands.registerCommand('automata.minimize', () => {
        vscode.window.showInformationMessage('Minimization feature coming soon!');
    }));
    context.subscriptions.push(vscode.commands.registerCommand('automata.convert', () => {
        vscode.window.showInformationMessage('Conversion feature coming soon!');
    }));
    context.subscriptions.push(vscode.commands.registerCommand('automata.validate', () => {
        vscode.window.showInformationMessage('Validation feature coming soon!');
    }));
    context.subscriptions.push(vscode.commands.registerCommand('automata.export', () => {
        vscode.window.showInformationMessage('Export feature coming soon!');
    }));
    context.subscriptions.push(vscode.commands.registerCommand('automata.testInput', () => {
        vscode.window.showInformationMessage('Test input feature coming soon!');
    }));
    // Create new automata commands
    context.subscriptions.push(vscode.commands.registerCommand('automata.createDFA', () => createNewAutomata('dfa')));
    context.subscriptions.push(vscode.commands.registerCommand('automata.createNFA', () => createNewAutomata('nfa')));
    context.subscriptions.push(vscode.commands.registerCommand('automata.createTM', () => createNewAutomata('tm')));
}
exports.activate = activate;
function deactivate() {
    return undefined;
}
exports.deactivate = deactivate;
function isAutomataFile(document) {
    return ['automata-dfa', 'automata-nfa', 'automata-tm'].includes(document.languageId) ||
        document.fileName.endsWith('.dfa') ||
        document.fileName.endsWith('.nfa') ||
        document.fileName.endsWith('.tm');
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
        const document = await vscode.workspace.openTextDocument({
            content: templates[type],
            language: 'json'
        });
        await vscode.window.showTextDocument(document);
    }
}
//# sourceMappingURL=extension-simple.js.map