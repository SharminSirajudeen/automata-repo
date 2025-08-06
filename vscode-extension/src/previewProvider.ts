import * as vscode from 'vscode';
import { AutomataParser, Automaton } from './parser';

export class AutomataPreviewProvider implements vscode.WebviewPanelSerializer {
    public static readonly viewType = 'automataPreview';
    private static currentPanel: vscode.WebviewPanel | undefined;
    private readonly _extensionUri: vscode.Uri;
    private readonly parser = new AutomataParser();

    constructor(extensionUri: vscode.Uri) {
        this._extensionUri = extensionUri;
    }

    public deserializeWebviewPanel(webviewPanel: vscode.WebviewPanel, state: any): Thenable<void> {
        // Handle restoring a panel that was previously serialized
        this.setupWebviewPanel(webviewPanel);
        return Promise.resolve();
    }

    public showPreview(uri: vscode.Uri, viewColumn?: vscode.ViewColumn) {
        const column = viewColumn || vscode.ViewColumn.Beside;

        // If we already have a panel, show it
        if (AutomataPreviewProvider.currentPanel) {
            AutomataPreviewProvider.currentPanel.reveal(column);
            this.updateContent(uri);
            return;
        }

        // Otherwise, create a new panel
        const panel = vscode.window.createWebviewPanel(
            AutomataPreviewProvider.viewType,
            'Automata Preview',
            column,
            this.getWebviewOptions()
        );

        AutomataPreviewProvider.currentPanel = panel;
        this.setupWebviewPanel(panel);
        this.updateContent(uri);
    }

    public updatePreview(uri: vscode.Uri) {
        if (AutomataPreviewProvider.currentPanel) {
            this.updateContent(uri);
        }
    }

    private setupWebviewPanel(panel: vscode.WebviewPanel) {
        panel.onDidDispose(() => {
            AutomataPreviewProvider.currentPanel = undefined;
        });

        // Handle messages from the webview
        panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'simulate':
                        this.handleSimulation(message.input);
                        break;
                    case 'step':
                        this.handleStep();
                        break;
                    case 'reset':
                        this.handleReset();
                        break;
                    case 'export':
                        this.handleExport(message.format);
                        break;
                }
            }
        );
    }

    private async updateContent(uri: vscode.Uri) {
        if (!AutomataPreviewProvider.currentPanel) return;

        try {
            const document = await vscode.workspace.openTextDocument(uri);
            const text = document.getText();
            const type = this.getAutomataType(uri);
            const automaton = this.parser.parse(text, type);
            
            const html = this.generateHtml(automaton, document.fileName);
            AutomataPreviewProvider.currentPanel.webview.html = html;
            AutomataPreviewProvider.currentPanel.title = `Preview: ${document.fileName}`;
        } catch (error) {
            const errorHtml = this.generateErrorHtml(error instanceof Error ? error.message : 'Unknown error');
            AutomataPreviewProvider.currentPanel.webview.html = errorHtml;
        }
    }

    private getAutomataType(uri: vscode.Uri): 'dfa' | 'nfa' | 'tm' {
        if (uri.fsPath.endsWith('.dfa')) return 'dfa';
        if (uri.fsPath.endsWith('.nfa')) return 'nfa';
        if (uri.fsPath.endsWith('.tm')) return 'tm';
        return 'dfa';
    }

    private generateHtml(automaton: Automaton, fileName: string): string {
        const scriptUri = AutomataPreviewProvider.currentPanel!.webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'preview.js')
        );
        const styleUri = AutomataPreviewProvider.currentPanel!.webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'preview.css')
        );

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Automata Preview - ${fileName}</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1>Automata Preview</h1>
                    <div class="automata-info">
                        <span class="type-badge ${automaton.type}">${automaton.type.toUpperCase()}</span>
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
                        
                        <div id="automata-diagram" class="diagram-container"></div>
                        
                        <div id="simulation-status" class="status-panel hidden">
                            <div class="status-info">
                                <span id="current-state" class="current-state"></span>
                                <span id="input-position" class="input-position"></span>
                                <span id="simulation-result" class="result"></span>
                            </div>
                            ${automaton.type === 'tm' ? '<div id="tape-view" class="tape-container"></div>' : ''}
                        </div>
                    </div>
                    
                    <div class="details-panel">
                        <div class="automata-details">
                            <h3>Automaton Details</h3>
                            <div class="detail-section">
                                <h4>States (${automaton.states.length})</h4>
                                <div class="state-list">
                                    ${automaton.states.map(state => `
                                        <span class="state-item ${automaton.accept_states.includes(state) ? 'accepting' : ''} ${state === automaton.start_state ? 'start' : ''}">
                                            ${state}
                                        </span>
                                    `).join('')}
                                </div>
                            </div>
                            
                            <div class="detail-section">
                                <h4>Alphabet</h4>
                                <div class="alphabet-list">
                                    ${automaton.alphabet.map(symbol => `<span class="symbol-item">${symbol}</span>`).join('')}
                                </div>
                            </div>
                            
                            ${automaton.type === 'tm' ? `
                                <div class="detail-section">
                                    <h4>Tape Alphabet</h4>
                                    <div class="alphabet-list">
                                        ${(automaton as any).tape_alphabet.map((symbol: string) => `<span class="symbol-item ${symbol === (automaton as any).blank_symbol ? 'blank' : ''}">${symbol}</span>`).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            <div class="detail-section">
                                <h4>Transitions</h4>
                                <div class="transitions-list">
                                    ${this.generateTransitionList(automaton)}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                const automaton = ${JSON.stringify(automaton)};
                
                // Initialize visualization
                initializeVisualization(automaton);
                
                // Event handlers
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
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }

    private generateTransitionList(automaton: Automaton): string {
        const transitions: string[] = [];
        
        for (const [state, stateTransitions] of Object.entries(automaton.transitions)) {
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                switch (automaton.type) {
                    case 'dfa':
                        transitions.push(`<div class="transition-item">δ(${state}, ${symbol}) = ${destination}</div>`);
                        break;
                    case 'nfa':
                        const destinations = destination as string[];
                        transitions.push(`<div class="transition-item">δ(${state}, ${symbol}) = {${destinations.join(', ')}}</div>`);
                        break;
                    case 'tm':
                        const tmTransition = destination as any;
                        transitions.push(`<div class="transition-item">δ(${state}, ${symbol}) = (${tmTransition.state}, ${tmTransition.write}, ${tmTransition.move})</div>`);
                        break;
                }
            }
        }
        
        return transitions.join('');
    }

    private generateErrorHtml(error: string): string {
        const styleUri = AutomataPreviewProvider.currentPanel!.webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'preview.css')
        );

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
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

    private getWebviewOptions(): vscode.WebviewOptions & vscode.WebviewPanelOptions {
        return {
            enableScripts: true,
            enableCommandUris: true,
            retainContextWhenHidden: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this._extensionUri, 'media')
            ]
        };
    }

    private handleSimulation(input: string) {
        // Handle simulation logic
        vscode.window.showInformationMessage(`Simulating with input: ${input}`);
    }

    private handleStep() {
        // Handle step logic
        vscode.window.showInformationMessage('Stepping...');
    }

    private handleReset() {
        // Handle reset logic
        vscode.window.showInformationMessage('Reset simulation');
    }

    private handleExport(format: string) {
        // Handle export logic
        vscode.window.showInformationMessage(`Exporting as ${format}...`);
    }
}