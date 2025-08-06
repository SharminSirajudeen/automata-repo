import * as vscode from 'vscode';

export class AutomataPreviewProvider implements vscode.WebviewPanelSerializer {
    public static readonly viewType = 'automataPreview';
    private static currentPanel: vscode.WebviewPanel | undefined;
    private readonly _extensionUri: vscode.Uri;

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
            }
        );
    }

    private async updateContent(uri: vscode.Uri) {
        if (!AutomataPreviewProvider.currentPanel) return;

        try {
            const document = await vscode.workspace.openTextDocument(uri);
            const text = document.getText();
            
            // Try to parse as JSON
            let automaton;
            try {
                automaton = JSON.parse(text);
            } catch (error) {
                throw new Error('Invalid JSON format');
            }
            
            const html = this.generateHtml(automaton, document.fileName);
            AutomataPreviewProvider.currentPanel.webview.html = html;
            AutomataPreviewProvider.currentPanel.title = `Preview: ${document.fileName}`;
        } catch (error) {
            const errorHtml = this.generateErrorHtml(error instanceof Error ? error.message : 'Unknown error');
            AutomataPreviewProvider.currentPanel.webview.html = errorHtml;
        }
    }

    private generateHtml(automaton: any, fileName: string): string {
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
                                    ${automaton.states ? automaton.states.map((state: string) => `
                                        <span class="state-item">${state}</span>
                                    `).join('') : 'No states defined'}
                                </div>
                            </div>
                            
                            <div class="detail-section">
                                <h4>Alphabet</h4>
                                <div class="alphabet-list">
                                    ${automaton.alphabet ? automaton.alphabet.map((symbol: string) => `<span class="symbol-item">${symbol}</span>`).join('') : 'No alphabet defined'}
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

    private generateErrorHtml(error: string): string {
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
}