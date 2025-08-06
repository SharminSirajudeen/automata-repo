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
exports.AutomataLanguageClient = void 0;
const path = __importStar(require("path"));
const vscode_1 = require("vscode");
const node_1 = require("vscode-languageclient/node");
class AutomataLanguageClient {
    constructor(context) {
        this.context = context;
        this.client = this.createLanguageClient();
    }
    createLanguageClient() {
        // The server is implemented in TypeScript and compiled to JavaScript
        const serverModule = this.context.asAbsolutePath(path.join('out', 'languageServer.js'));
        // If the extension is launched in debug mode then the debug server options are used
        // Otherwise the run options are used
        const serverOptions = {
            run: { module: serverModule, transport: node_1.TransportKind.ipc },
            debug: {
                module: serverModule,
                transport: node_1.TransportKind.ipc,
                options: { execArgv: ['--nolazy', '--inspect=6009'] }
            }
        };
        // Options to control the language client
        const clientOptions = {
            // Register the server for automata documents
            documentSelector: [
                { scheme: 'file', language: 'automata-dfa' },
                { scheme: 'file', language: 'automata-nfa' },
                { scheme: 'file', language: 'automata-tm' }
            ],
            synchronize: {
                // Notify the server about file changes to automata files
                fileEvents: vscode_1.workspace.createFileSystemWatcher('**/*.{dfa,nfa,tm}')
            }
        };
        // Create the language client and return it
        return new node_1.LanguageClient('automataLanguageServer', 'Automata Language Server', serverOptions, clientOptions);
    }
    start() {
        this.client.start();
    }
    stop() {
        if (!this.client) {
            return Promise.resolve();
        }
        return this.client.stop();
    }
    getClient() {
        return this.client;
    }
}
exports.AutomataLanguageClient = AutomataLanguageClient;
//# sourceMappingURL=languageClient.js.map