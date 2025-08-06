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
exports.AutomataPreviewProvider = void 0;
const vscode = __importStar(require("vscode"));
class AutomataPreviewProvider {
    constructor(extensionUri) {
        this._extensionUri = extensionUri;
    }
    deserializeWebviewPanel(webviewPanel, state) {
        // Handle restoring a panel that was previously serialized
        this.setupWebviewPanel(webviewPanel);
        return Promise.resolve();
    }
    showPreview(uri, viewColumn) {
        const column = viewColumn || vscode.ViewColumn.Beside;
        // If we already have a panel, show it
        if (AutomataPreviewProvider.currentPanel) {
            AutomataPreviewProvider.currentPanel.reveal(column);
            this.updateContent(uri);
            return;
        }
        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel(AutomataPreviewProvider.viewType, 'Automata Preview', column, this.getWebviewOptions());
        AutomataPreviewProvider.currentPanel = panel;
        this.setupWebviewPanel(panel);
        this.updateContent(uri);
    }
    updatePreview(uri) {
        if (AutomataPreviewProvider.currentPanel) {
            this.updateContent(uri);
        }
    }
    setupWebviewPanel(panel) {
        panel.onDidDispose(() => {
            AutomataPreviewProvider.currentPanel = undefined;
        });
        // Handle messages from the webview
        panel.webview.onDidReceiveMessage(message => {
            switch (message.command) {
                case 'simulate':
                    vscode.window.showInformationMessage(`Simulating with input: ${message.input}`);
                    break;
                case 'step':
                    vscode.window.showInformationMessage('Stepping...');
                    break;
                case 'reset':
                    vscode.window.showInformationMessage('Reset simulation');
                    break;
                case 'export':
                    vscode.window.showInformationMessage(`Exporting as ${message.format}...`);
                    break;
            }
        });
    }
    async updateContent(uri) {
        if (!AutomataPreviewProvider.currentPanel)
            return;
        try {
            const document = await vscode.workspace.openTextDocument(uri);
            const text = document.getText();
            // Try to parse as JSON
            let automaton;
            try {
                automaton = JSON.parse(text);
            }
            catch (error) {
                throw new Error('Invalid JSON format');
            }
            const html = this.generateHtml(automaton, document.fileName);
            AutomataPreviewProvider.currentPanel.webview.html = html;
            AutomataPreviewProvider.currentPanel.title = `Preview: ${document.fileName}`;
        }
        catch (error) {
            const errorHtml = this.generateErrorHtml(error instanceof Error ? error.message : 'Unknown error');
            AutomataPreviewProvider.currentPanel.webview.html = errorHtml;
        }
    }
    generateHtml(automaton, fileName) {
        const styleUri = AutomataPreviewProvider.currentPanel.webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'preview.css'));
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Automata Preview - ${fileName}</title>
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1>Automata Preview</h1>
                    <div class="automata-info">
                        <span class="type-badge">${automaton.type || 'Unknown'}</span>
                        <span class="filename">${fileName}</span>
                    </div>
                </header>
                
                <div class="main-content">
                    <div class="visualization-panel">
                        <div class="controls">
                            <div class="simulation-controls">
                                <input type="text" id="input-string" placeholder="Enter input string..." />
                                <button id="simulate-btn" class="btn primary">Simulate</button>
                                <button id="step-btn" class="btn secondary" disabled>Step</button>
                                <button id="reset-btn" class="btn secondary" disabled>Reset</button>
                            </div>
                            <div class="export-controls">
                                <select id="export-format">
                                    <option value="svg">SVG</option>
                                    <option value="png">PNG</option>
                                    <option value="json">JSON</option>
                                    <option value="dot">DOT</option>
                                </select>
                                <button id="export-btn" class="btn secondary">Export</button>
                            </div>
                        </div>
                        
                        <div id="automata-diagram" class="diagram-container">
                            <p>Automata visualization will appear here</p>
                            <pre>${JSON.stringify(automaton, null, 2)}</pre>
                        </div>
                    </div>
                    
                    <div class="details-panel">
                        <div class="automata-details">
                            <h3>Automaton Details</h3>
                            <div class="detail-section">
                                <h4>States</h4>
                                <div class="state-list">
                                    ${automaton.states ? automaton.states.map((state) => `
                                        <span class="state-item">${state}</span>
                                    `).join('') : 'No states defined'}
                                </div>
                            </div>
                            
                            <div class="detail-section">
                                <h4>Alphabet</h4>
                                <div class="alphabet-list">
                                    ${automaton.alphabet ? automaton.alphabet.map((symbol) => `<span class="symbol-item">${symbol}</span>`).join('') : 'No alphabet defined'}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                
                document.getElementById('simulate-btn').addEventListener('click', () => {
                    const input = document.getElementById('input-string').value;
                    vscode.postMessage({ command: 'simulate', input: input });
                });
                
                document.getElementById('step-btn').addEventListener('click', () => {
                    vscode.postMessage({ command: 'step' });
                });
                
                document.getElementById('reset-btn').addEventListener('click', () => {
                    vscode.postMessage({ command: 'reset' });
                });
                
                document.getElementById('export-btn').addEventListener('click', () => {
                    const format = document.getElementById('export-format').value;
                    vscode.postMessage({ command: 'export', format: format });
                });
            </script>
        </body>
        </html>`;
    }
    generateErrorHtml(error) {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Automata Preview - Error</title>
        </head>
        <body>
            <div class="container">
                <div class="error-container">
                    <h2>Preview Error</h2>
                    <p class="error-message">${error}</p>
                    <p class="error-help">Please check your automaton definition and try again.</p>
                </div>
            </div>
        </body>
        </html>`;
    }
    getWebviewOptions() {
        return {
            enableScripts: true,
            enableCommandUris: true,
            retainContextWhenHidden: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this._extensionUri, 'media')
            ]
        };
    }
}
exports.AutomataPreviewProvider = AutomataPreviewProvider;
AutomataPreviewProvider.viewType = 'automataPreview';
//# sourceMappingURL=previewProvider-simple.js.map