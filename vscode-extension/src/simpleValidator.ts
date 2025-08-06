import * as vscode from 'vscode';

export class AutomataValidator {
    async validateDocument(document: vscode.TextDocument): Promise<void> {
        // Simple validation - just check if it's valid JSON
        try {
            JSON.parse(document.getText());
            // If we get here, JSON is valid
        } catch (error) {
            // Show error for invalid JSON
            vscode.window.showErrorMessage('Invalid JSON in automata file');
        }
    }

    validate(automaton: any, document: vscode.TextDocument): any[] {
        // Return empty array for now
        return [];
    }
}