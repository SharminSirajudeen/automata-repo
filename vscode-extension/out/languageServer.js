"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const node_1 = require("vscode-languageserver/node");
const vscode_languageserver_textdocument_1 = require("vscode-languageserver-textdocument");
// Create a connection for the server
const connection = (0, node_1.createConnection)(node_1.ProposedFeatures.all);
// Create a simple text document manager
const documents = new node_1.TextDocuments(vscode_languageserver_textdocument_1.TextDocument);
connection.onInitialize((params) => {
    const result = {
        capabilities: {
            textDocumentSync: node_1.TextDocumentSyncKind.Incremental,
            completionProvider: {
                resolveProvider: true,
                triggerCharacters: ['"', ':', '{', '[']
            },
            hoverProvider: true
        }
    };
    return result;
});
connection.onInitialized(() => {
    connection.console.log('Automata Language Server initialized');
});
// Provide completion items
connection.onCompletion((textDocumentPosition) => {
    const document = documents.get(textDocumentPosition.textDocument.uri);
    if (!document)
        return [];
    return getCompletionItems();
});
function getCompletionItems() {
    const items = [];
    const keywords = [
        { label: 'states', insertText: '"states": []', detail: 'Define the set of states' },
        { label: 'alphabet', insertText: '"alphabet": []', detail: 'Define the input alphabet' },
        { label: 'start_state', insertText: '"start_state": ""', detail: 'Define the initial state' },
        { label: 'accept_states', insertText: '"accept_states": []', detail: 'Define accepting states' },
        { label: 'transitions', insertText: '"transitions": {}', detail: 'Define state transitions' }
    ];
    keywords.forEach(keyword => {
        items.push({
            label: keyword.label,
            kind: 14,
            data: keyword.detail,
            insertText: keyword.insertText
        });
    });
    return items;
}
connection.onCompletionResolve((item) => {
    if (item.data) {
        item.detail = item.data;
    }
    return item;
});
// Provide hover information
connection.onHover((textDocumentPosition) => {
    const document = documents.get(textDocumentPosition.textDocument.uri);
    if (!document)
        return null;
    return {
        contents: {
            kind: node_1.MarkupKind.Markdown,
            value: 'Automata definition hover information'
        }
    };
});
documents.onDidChangeContent(() => {
    // Simple validation could go here
});
// Make the text document manager listen on the connection
documents.listen(connection);
// Listen on the connection
connection.listen();
//# sourceMappingURL=languageServer.js.map