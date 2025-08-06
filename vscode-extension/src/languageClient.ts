import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

export class AutomataLanguageClient {
    private client: LanguageClient;

    constructor(private context: ExtensionContext) {
        this.client = this.createLanguageClient();
    }

    private createLanguageClient(): LanguageClient {
        // The server is implemented in TypeScript and compiled to JavaScript
        const serverModule = this.context.asAbsolutePath(
            path.join('out', 'languageServer.js')
        );

        // If the extension is launched in debug mode then the debug server options are used
        // Otherwise the run options are used
        const serverOptions: ServerOptions = {
            run: { module: serverModule, transport: TransportKind.ipc },
            debug: {
                module: serverModule,
                transport: TransportKind.ipc,
                options: { execArgv: ['--nolazy', '--inspect=6009'] }
            }
        };

        // Options to control the language client
        const clientOptions: LanguageClientOptions = {
            // Register the server for automata documents
            documentSelector: [
                { scheme: 'file', language: 'automata-dfa' },
                { scheme: 'file', language: 'automata-nfa' },
                { scheme: 'file', language: 'automata-tm' }
            ],
            synchronize: {
                // Notify the server about file changes to automata files
                fileEvents: workspace.createFileSystemWatcher('**/*.{dfa,nfa,tm}')
            }
        };

        // Create the language client and return it
        return new LanguageClient(
            'automataLanguageServer',
            'Automata Language Server',
            serverOptions,
            clientOptions
        );
    }

    public start(): void {
        this.client.start();
    }

    public stop(): Thenable<void> {
        if (!this.client) {
            return Promise.resolve();
        }
        return this.client.stop();
    }

    public getClient(): LanguageClient {
        return this.client;
    }
}