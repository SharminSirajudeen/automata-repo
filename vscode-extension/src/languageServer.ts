import {
    createConnection,
    TextDocuments,
    ProposedFeatures,
    InitializeParams,
    CompletionItem,
    TextDocumentPositionParams,
    TextDocumentSyncKind,
    InitializeResult,
    Hover,
    MarkupKind,
    DiagnosticSeverity
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';

// Create a connection for the server
const connection = createConnection(ProposedFeatures.all);

// Create a simple text document manager
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

connection.onInitialize((params: InitializeParams) => {
    const result: InitializeResult = {
        capabilities: {
            textDocumentSync: TextDocumentSyncKind.Incremental,
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
connection.onCompletion(
    (textDocumentPosition: TextDocumentPositionParams): CompletionItem[] => {
        const document = documents.get(textDocumentPosition.textDocument.uri);
        if (!document) return [];

        return getCompletionItems();
    }
);

function getCompletionItems(): CompletionItem[] {
    const items: CompletionItem[] = [];
    
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
            kind: 14, // Keyword
            data: keyword.detail,
            insertText: keyword.insertText
        });
    });

    return items;
}

connection.onCompletionResolve(
    (item: CompletionItem): CompletionItem => {
        if (item.data) {
            item.detail = item.data;
        }
        return item;
    }
);

// Provide hover information
connection.onHover(
    (textDocumentPosition: TextDocumentPositionParams): Hover | null => {
        const document = documents.get(textDocumentPosition.textDocument.uri);
        if (!document) return null;

        return {
            contents: {
                kind: MarkupKind.Markdown,
                value: 'Automata definition hover information'
            }
        };
    }
);

documents.onDidChangeContent(() => {
    // Simple validation could go here
});

// Make the text document manager listen on the connection
documents.listen(connection);

// Listen on the connection
connection.listen();