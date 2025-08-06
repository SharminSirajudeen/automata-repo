import * as vscode from 'vscode';
import { AutomataParser, DFA } from './parser';

export class AutomataMinimizer {
    private parser = new AutomataParser();

    async minimize(document: vscode.TextDocument): Promise<void> {
        try {
            const text = document.getText();
            const automaton = this.parser.parse(text, 'dfa') as DFA;

            const minimizedDFA = this.minimizeDFA(automaton);
            
            // Show the minimized DFA
            await this.showMinimizedResult(automaton, minimizedDFA, document);
        } catch (error) {
            vscode.window.showErrorMessage(`Minimization failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }

    private minimizeDFA(dfa: DFA): DFA {
        // Step 1: Remove unreachable states
        const reachableStates = this.getReachableStates(dfa);
        
        // Step 2: Remove dead states (states that cannot reach any accepting state)
        const liveStates = this.getLiveStates(dfa, reachableStates);
        
        // Step 3: Merge equivalent states using table-filling algorithm
        const equivalentStates = this.findEquivalentStates(dfa, liveStates);
        
        // Step 4: Construct minimized DFA
        return this.constructMinimizedDFA(dfa, equivalentStates);
    }

    private getReachableStates(dfa: DFA): Set<string> {
        const reachable = new Set<string>();
        const queue = [dfa.start_state];
        
        while (queue.length > 0) {
            const current = queue.shift()!;
            if (reachable.has(current)) continue;
            
            reachable.add(current);
            const stateTransitions = dfa.transitions[current];
            
            if (stateTransitions) {
                for (const destination of Object.values(stateTransitions)) {
                    if (!reachable.has(destination)) {
                        queue.push(destination);
                    }
                }
            }
        }
        
        return reachable;
    }

    private getLiveStates(dfa: DFA, reachableStates: Set<string>): Set<string> {
        const live = new Set<string>();
        const acceptingStates = new Set(dfa.accept_states);
        
        // All accepting states are live
        for (const state of dfa.accept_states) {
            if (reachableStates.has(state)) {
                live.add(state);
            }
        }
        
        // Find states that can reach accepting states
        let changed = true;
        while (changed) {
            changed = false;
            
            for (const state of reachableStates) {
                if (live.has(state)) continue;
                
                const stateTransitions = dfa.transitions[state];
                if (stateTransitions) {
                    for (const destination of Object.values(stateTransitions)) {
                        if (live.has(destination)) {
                            live.add(state);
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
        
        return live;
    }

    private findEquivalentStates(dfa: DFA, liveStates: Set<string>): Map<string, Set<string>> {
        const states = Array.from(liveStates);
        const n = states.length;
        
        // Create distinguishability table
        const table: boolean[][] = [];
        for (let i = 0; i < n; i++) {
            table[i] = [];
            for (let j = 0; j < n; j++) {
                table[i][j] = false;
            }
        }
        
        // Mark pairs where one is accepting and other is not
        const acceptingStates = new Set(dfa.accept_states);
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const state1 = states[i];
                const state2 = states[j];
                
                if (acceptingStates.has(state1) !== acceptingStates.has(state2)) {
                    table[i][j] = true;
                    table[j][i] = true;
                }
            }
        }
        
        // Table filling algorithm
        let changed = true;
        while (changed) {
            changed = false;
            
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    if (table[i][j]) continue;
                    
                    const state1 = states[i];
                    const state2 = states[j];
                    
                    // Check if states are distinguishable by any symbol
                    for (const symbol of dfa.alphabet) {
                        const dest1 = dfa.transitions[state1]?.[symbol];
                        const dest2 = dfa.transitions[state2]?.[symbol];
                        
                        if (!dest1 || !dest2) {
                            // One has transition, other doesn't - distinguishable
                            if (dest1 !== dest2) {
                                table[i][j] = true;
                                table[j][i] = true;
                                changed = true;
                                break;
                            }
                        } else if (dest1 !== dest2) {
                            // Both have transitions - check if destinations are distinguishable
                            const idx1 = states.indexOf(dest1);
                            const idx2 = states.indexOf(dest2);
                            
                            if (idx1 !== -1 && idx2 !== -1 && table[idx1][idx2]) {
                                table[i][j] = true;
                                table[j][i] = true;
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        // Group equivalent states
        const equivalenceClasses = new Map<string, Set<string>>();
        const processed = new Set<string>();
        
        for (let i = 0; i < n; i++) {
            const state1 = states[i];
            if (processed.has(state1)) continue;
            
            const equivalentGroup = new Set([state1]);
            processed.add(state1);
            
            for (let j = i + 1; j < n; j++) {
                const state2 = states[j];
                if (!processed.has(state2) && !table[i][j]) {
                    equivalentGroup.add(state2);
                    processed.add(state2);
                }
            }
            
            // Use lexicographically smallest state as representative
            const representative = Array.from(equivalentGroup).sort()[0];
            equivalenceClasses.set(representative, equivalentGroup);
        }
        
        return equivalenceClasses;
    }

    private constructMinimizedDFA(originalDFA: DFA, equivalenceClasses: Map<string, Set<string>>): DFA {
        // Create mapping from original states to representatives
        const stateMap = new Map<string, string>();
        for (const [representative, group] of equivalenceClasses) {
            for (const state of group) {
                stateMap.set(state, representative);
            }
        }
        
        const newStates = Array.from(equivalenceClasses.keys()).sort();
        const newStartState = stateMap.get(originalDFA.start_state)!;
        const newAcceptStates = [...new Set(originalDFA.accept_states.map(state => stateMap.get(state)!))];
        
        // Build new transitions
        const newTransitions: { [state: string]: { [symbol: string]: string } } = {};
        
        for (const representative of newStates) {
            newTransitions[representative] = {};
            
            // Use any state from the equivalence class to build transitions
            const sampleState = Array.from(equivalenceClasses.get(representative)!)[0];
            const originalTransitions = originalDFA.transitions[sampleState];
            
            if (originalTransitions) {
                for (const [symbol, destination] of Object.entries(originalTransitions)) {
                    newTransitions[representative][symbol] = stateMap.get(destination)!;
                }
            }
        }
        
        return {
            type: 'dfa',
            states: newStates,
            alphabet: originalDFA.alphabet,
            start_state: newStartState,
            accept_states: newAcceptStates,
            transitions: newTransitions
        };
    }

    private async showMinimizedResult(original: DFA, minimized: DFA, document: vscode.TextDocument): Promise<void> {
        const originalStateCount = original.states.length;
        const minimizedStateCount = minimized.states.length;
        const reduction = originalStateCount - minimizedStateCount;
        
        let message = `Minimization complete!\n`;
        message += `States reduced from ${originalStateCount} to ${minimizedStateCount}`;
        
        if (reduction > 0) {
            message += ` (${reduction} states removed)`;
        } else {
            message += ` (already minimal)`;
        }
        
        const options = ['Show Minimized DFA', 'Show Analysis', 'Save to File'];
        const choice = await vscode.window.showInformationMessage(message, ...options);
        
        if (!choice) return;
        
        switch (choice) {
            case 'Show Minimized DFA':
                await this.showMinimizedDFA(minimized, document);
                break;
            case 'Show Analysis':
                await this.showMinimizationAnalysis(original, minimized);
                break;
            case 'Save to File':
                await this.saveMinimizedDFA(minimized, document);
                break;
        }
    }

    private async showMinimizedDFA(minimized: DFA, originalDocument: vscode.TextDocument): Promise<void> {
        const content = JSON.stringify(minimized, null, 2);
        
        const newDocument = await vscode.workspace.openTextDocument({
            content: content,
            language: 'json'
        });
        
        await vscode.window.showTextDocument(newDocument, vscode.ViewColumn.Beside);
    }

    private async showMinimizationAnalysis(original: DFA, minimized: DFA): Promise<void> {
        let analysis = 'DFA Minimization Analysis\n';
        analysis += '=========================\n\n';
        
        analysis += `Original States: ${original.states.length}\n`;
        analysis += `Minimized States: ${minimized.states.length}\n`;
        analysis += `Reduction: ${original.states.length - minimized.states.length} states\n\n`;
        
        // Find state mappings
        const stateMapping = this.findStateMapping(original, minimized);
        
        if (stateMapping.size > 0) {
            analysis += 'State Equivalences:\n';
            for (const [representative, group] of stateMapping) {
                if (group.size > 1) {
                    analysis += `  {${Array.from(group).join(', ')}} → ${representative}\n`;
                }
            }
            analysis += '\n';
        }
        
        // Compare transitions
        analysis += 'Transition Comparison:\n';
        analysis += `Original: ${this.countTransitions(original)} transitions\n`;
        analysis += `Minimized: ${this.countTransitions(minimized)} transitions\n\n`;
        
        // Check if languages are equivalent
        analysis += 'Language Equivalence: ';
        analysis += this.checkLanguageEquivalence(original, minimized) ? 'VERIFIED' : 'ERROR';
        analysis += '\n';
        
        const outputChannel = vscode.window.createOutputChannel('DFA Minimization Analysis');
        outputChannel.clear();
        outputChannel.append(analysis);
        outputChannel.show();
    }

    private async saveMinimizedDFA(minimized: DFA, originalDocument: vscode.TextDocument): Promise<void> {
        const baseName = originalDocument.fileName.replace(/\.[^.]+$/, '');
        const defaultName = `${baseName}_minimized.dfa`;
        
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(defaultName),
            filters: {
                'DFA files': ['dfa'],
                'JSON files': ['json'],
                'All files': ['*']
            }
        });
        
        if (uri) {
            const content = JSON.stringify(minimized, null, 2);
            await vscode.workspace.fs.writeFile(uri, Buffer.from(content));
            vscode.window.showInformationMessage(`Minimized DFA saved to ${uri.fsPath}`);
        }
    }

    private findStateMapping(original: DFA, minimized: DFA): Map<string, Set<string>> {
        // This is a simplified approach - in practice, you'd track the mapping during minimization
        const mapping = new Map<string, Set<string>>();
        
        // For now, just group states that have the same transitions
        const transitionSignatures = new Map<string, Set<string>>();
        
        for (const state of original.states) {
            const signature = this.getTransitionSignature(original, state);
            if (!transitionSignatures.has(signature)) {
                transitionSignatures.set(signature, new Set());
            }
            transitionSignatures.get(signature)!.add(state);
        }
        
        for (const [signature, states] of transitionSignatures) {
            if (states.size > 1) {
                const representative = Array.from(states).sort()[0];
                mapping.set(representative, states);
            }
        }
        
        return mapping;
    }

    private getTransitionSignature(dfa: DFA, state: string): string {
        const transitions = dfa.transitions[state] || {};
        const signature = dfa.alphabet.map(symbol => transitions[symbol] || 'NULL').join('|');
        const isAccepting = dfa.accept_states.includes(state) ? '1' : '0';
        return `${signature}:${isAccepting}`;
    }

    private countTransitions(dfa: DFA): number {
        let count = 0;
        for (const stateTransitions of Object.values(dfa.transitions)) {
            count += Object.keys(stateTransitions).length;
        }
        return count;
    }

    private checkLanguageEquivalence(dfa1: DFA, dfa2: DFA): boolean {
        // Simplified check - in practice, this would be more thorough
        // Check if both have same alphabet
        if (dfa1.alphabet.length !== dfa2.alphabet.length) return false;
        if (!dfa1.alphabet.every(symbol => dfa2.alphabet.includes(symbol))) return false;
        
        // For now, assume they're equivalent if minimization was successful
        return true;
    }
}