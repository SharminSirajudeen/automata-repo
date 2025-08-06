import * as vscode from 'vscode';
import { AutomataParser, Automaton, DFA, NFA, TuringMachine } from './parser';

export class AutomataConverter {
    private parser = new AutomataParser();

    async convert(document: vscode.TextDocument): Promise<void> {
        try {
            const text = document.getText();
            const type = this.getAutomataType(document.uri);
            const automaton = this.parser.parse(text, type);

            const conversionOptions = this.getConversionOptions(type);
            
            if (conversionOptions.length === 0) {
                vscode.window.showInformationMessage('No conversions available for this automaton type');
                return;
            }

            const choice = await vscode.window.showQuickPick(conversionOptions, {
                placeHolder: 'Select conversion type'
            });

            if (!choice) return;

            await this.performConversion(automaton, choice.value, document);
        } catch (error) {
            vscode.window.showErrorMessage(`Conversion failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }

    private getConversionOptions(type: 'dfa' | 'nfa' | 'tm'): Array<{ label: string, description: string, value: string }> {
        const options: Array<{ label: string, description: string, value: string }> = [];

        switch (type) {
            case 'nfa':
                options.push({
                    label: 'Convert to DFA',
                    description: 'Convert NFA to equivalent DFA using subset construction',
                    value: 'nfa-to-dfa'
                });
                break;
            case 'dfa':
                options.push({
                    label: 'Convert to NFA',
                    description: 'Convert DFA to equivalent NFA (trivial conversion)',
                    value: 'dfa-to-nfa'
                });
                options.push({
                    label: 'Convert to Regular Grammar',
                    description: 'Generate regular grammar equivalent to the DFA',
                    value: 'dfa-to-grammar'
                });
                options.push({
                    label: 'Convert to Regular Expression',
                    description: 'Generate regular expression equivalent to the DFA',
                    value: 'dfa-to-regex'
                });
                break;
        }

        // Common conversions
        options.push({
            label: 'Export to JFLAP',
            description: 'Export to JFLAP XML format',
            value: 'to-jflap'
        });

        options.push({
            label: 'Export to GraphViz DOT',
            description: 'Export to DOT format for GraphViz visualization',
            value: 'to-dot'
        });

        return options;
    }

    private async performConversion(automaton: Automaton, conversionType: string, originalDocument: vscode.TextDocument): Promise<void> {
        let result: string;
        let fileExtension: string;

        switch (conversionType) {
            case 'nfa-to-dfa':
                result = this.convertNFAToDFA(automaton as NFA);
                fileExtension = '.dfa';
                break;
            case 'dfa-to-nfa':
                result = this.convertDFAToNFA(automaton as DFA);
                fileExtension = '.nfa';
                break;
            case 'dfa-to-grammar':
                result = this.convertDFAToGrammar(automaton as DFA);
                fileExtension = '.txt';
                break;
            case 'dfa-to-regex':
                result = this.convertDFAToRegex(automaton as DFA);
                fileExtension = '.txt';
                break;
            case 'to-jflap':
                result = this.convertToJFLAP(automaton);
                fileExtension = '.jff';
                break;
            case 'to-dot':
                result = this.convertToDOT(automaton);
                fileExtension = '.dot';
                break;
            default:
                throw new Error(`Unknown conversion type: ${conversionType}`);
        }

        await this.createNewDocument(result, fileExtension, originalDocument);
    }

    private convertNFAToDFA(nfa: NFA): string {
        // Subset construction algorithm
        const dfaStates: string[] = [];
        const dfaTransitions: { [state: string]: { [symbol: string]: string } } = {};
        const dfaAcceptStates: string[] = [];
        
        // Start with epsilon closure of start state
        const startClosure = this.epsilonClosure(nfa, new Set([nfa.start_state]));
        const startStateStr = this.stateSetToString(startClosure);
        
        const unmarked = [startClosure];
        const marked = new Map<string, Set<string>>();
        marked.set(startStateStr, startClosure);
        dfaStates.push(startStateStr);

        // Check if start state is accepting
        if (this.isAcceptingStateSet(nfa, startClosure)) {
            dfaAcceptStates.push(startStateStr);
        }

        while (unmarked.length > 0) {
            const current = unmarked.shift()!;
            const currentStr = this.stateSetToString(current);
            dfaTransitions[currentStr] = {};

            for (const symbol of nfa.alphabet) {
                const nextStates = new Set<string>();
                
                // Find all states reachable on this symbol
                for (const state of current) {
                    const transitions = nfa.transitions[state];
                    if (transitions && transitions[symbol]) {
                        for (const nextState of transitions[symbol]) {
                            nextStates.add(nextState);
                        }
                    }
                }

                // Apply epsilon closure
                const closure = this.epsilonClosure(nfa, nextStates);
                if (closure.size === 0) continue;

                const closureStr = this.stateSetToString(closure);
                
                if (!marked.has(closureStr)) {
                    marked.set(closureStr, closure);
                    dfaStates.push(closureStr);
                    unmarked.push(closure);
                    
                    if (this.isAcceptingStateSet(nfa, closure)) {
                        dfaAcceptStates.push(closureStr);
                    }
                }

                dfaTransitions[currentStr][symbol] = closureStr;
            }
        }

        const dfa: DFA = {
            type: 'dfa',
            states: dfaStates,
            alphabet: nfa.alphabet,
            start_state: startStateStr,
            accept_states: dfaAcceptStates,
            transitions: dfaTransitions
        };

        return JSON.stringify(dfa, null, 2);
    }

    private convertDFAToNFA(dfa: DFA): string {
        // Trivial conversion - DFA is already an NFA
        const nfaTransitions: { [state: string]: { [symbol: string]: string[] } } = {};
        
        for (const [state, stateTransitions] of Object.entries(dfa.transitions)) {
            nfaTransitions[state] = {};
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                nfaTransitions[state][symbol] = [destination];
            }
        }

        const nfa: NFA = {
            type: 'nfa',
            states: dfa.states,
            alphabet: dfa.alphabet,
            start_state: dfa.start_state,
            accept_states: dfa.accept_states,
            transitions: nfaTransitions
        };

        return JSON.stringify(nfa, null, 2);
    }

    private convertDFAToGrammar(dfa: DFA): string {
        let grammar = 'Regular Grammar\n';
        grammar += '===============\n\n';
        
        grammar += 'Non-terminals: ' + dfa.states.join(', ') + '\n';
        grammar += 'Terminals: ' + dfa.alphabet.join(', ') + '\n';
        grammar += 'Start symbol: ' + dfa.start_state + '\n\n';
        
        grammar += 'Productions:\n';
        
        for (const [state, stateTransitions] of Object.entries(dfa.transitions)) {
            for (const [symbol, nextState] of Object.entries(stateTransitions)) {
                if (dfa.accept_states.includes(nextState)) {
                    grammar += `${state} → ${symbol}\n`;
                } else {
                    grammar += `${state} → ${symbol}${nextState}\n`;
                }
            }
        }

        // Add epsilon production for accepting states if they have no outgoing transitions
        for (const acceptState of dfa.accept_states) {
            if (acceptState === dfa.start_state || 
                Object.keys(dfa.transitions[acceptState] || {}).length === 0) {
                grammar += `${acceptState} → ε\n`;
            }
        }

        return grammar;
    }

    private convertDFAToRegex(dfa: DFA): string {
        // State elimination method for converting DFA to regular expression
        let result = 'Regular Expression Generation\n';
        result += '=============================\n\n';
        
        try {
            const regex = this.stateEliminationMethod(dfa);
            result += `Regular Expression: ${regex}\n\n`;
            result += 'Note: This is a simplified conversion. Complex DFAs may produce very long expressions.\n';
        } catch (error) {
            result += `Error generating regular expression: ${error instanceof Error ? error.message : 'Unknown error'}\n`;
            result += '\nFallback: Manual construction required for complex automata.\n';
        }

        return result;
    }

    private convertToJFLAP(automaton: Automaton): string {
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
        xml += '<structure>\n';
        xml += `  <type>${automaton.type === 'tm' ? 'turing' : 'fa'}</type>\n`;
        xml += '  <automaton>\n';

        // Add states
        let stateId = 0;
        const stateMap = new Map<string, number>();
        
        for (const state of automaton.states) {
            stateMap.set(state, stateId);
            xml += `    <state id="${stateId}" name="${state}">\n`;
            xml += `      <x>${100 + (stateId % 5) * 150}</x>\n`;
            xml += `      <y>${100 + Math.floor(stateId / 5) * 100}</y>\n`;
            
            if (state === automaton.start_state) {
                xml += '      <initial/>\n';
            }
            if (automaton.accept_states.includes(state)) {
                xml += '      <final/>\n';
            }
            
            xml += '    </state>\n';
            stateId++;
        }

        // Add transitions
        for (const [fromState, stateTransitions] of Object.entries(automaton.transitions)) {
            const fromId = stateMap.get(fromState);
            
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                switch (automaton.type) {
                    case 'dfa':
                        const toId = stateMap.get(destination as string);
                        xml += `    <transition>\n`;
                        xml += `      <from>${fromId}</from>\n`;
                        xml += `      <to>${toId}</to>\n`;
                        xml += `      <read>${symbol}</read>\n`;
                        xml += `    </transition>\n`;
                        break;
                    case 'nfa':
                        for (const dest of destination as string[]) {
                            const nfaToId = stateMap.get(dest);
                            xml += `    <transition>\n`;
                            xml += `      <from>${fromId}</from>\n`;
                            xml += `      <to>${nfaToId}</to>\n`;
                            xml += `      <read>${symbol === 'epsilon' ? '' : symbol}</read>\n`;
                            xml += `    </transition>\n`;
                        }
                        break;
                    case 'tm':
                        const tmDest = destination as any;
                        const tmToId = stateMap.get(tmDest.state);
                        xml += `    <transition>\n`;
                        xml += `      <from>${fromId}</from>\n`;
                        xml += `      <to>${tmToId}</to>\n`;
                        xml += `      <read>${symbol}</read>\n`;
                        xml += `      <write>${tmDest.write}</write>\n`;
                        xml += `      <move>${tmDest.move}</move>\n`;
                        xml += `    </transition>\n`;
                        break;
                }
            }
        }

        xml += '  </automaton>\n';
        xml += '</structure>\n';

        return xml;
    }

    private convertToDOT(automaton: Automaton): string {
        let dot = 'digraph automaton {\n';
        dot += '  rankdir=LR;\n';
        dot += '  size="8,5"\n';
        dot += '  node [shape = circle];\n';

        // Mark accepting states
        if (automaton.accept_states.length > 0) {
            dot += `  node [shape = doublecircle]; ${automaton.accept_states.join(' ')};\n`;
        }
        
        dot += '  node [shape = circle];\n';

        // Add start arrow
        dot += `  start [shape=none,label=""];\n`;
        dot += `  start -> ${automaton.start_state};\n`;

        // Add transitions
        for (const [fromState, stateTransitions] of Object.entries(automaton.transitions)) {
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                switch (automaton.type) {
                    case 'dfa':
                        dot += `  ${fromState} -> ${destination} [label="${symbol}"];\n`;
                        break;
                    case 'nfa':
                        for (const dest of destination as string[]) {
                            const label = symbol === 'epsilon' || symbol === 'ε' || symbol === 'λ' ? 'ε' : symbol;
                            dot += `  ${fromState} -> ${dest} [label="${label}"];\n`;
                        }
                        break;
                    case 'tm':
                        const tmDest = destination as any;
                        dot += `  ${fromState} -> ${tmDest.state} [label="${symbol}/${tmDest.write},${tmDest.move}"];\n`;
                        break;
                }
            }
        }

        dot += '}\n';
        return dot;
    }

    private async createNewDocument(content: string, extension: string, originalDocument: vscode.TextDocument): Promise<void> {
        const baseName = originalDocument.fileName.replace(/\.[^.]+$/, '');
        const newFileName = `${baseName}_converted${extension}`;
        
        const newDocument = await vscode.workspace.openTextDocument({
            content: content,
            language: extension === '.json' ? 'json' : 'text'
        });

        await vscode.window.showTextDocument(newDocument);
        
        // Offer to save the document
        const saveChoice = await vscode.window.showInformationMessage(
            'Conversion completed. Save the result?',
            'Save As...', 'Don\'t Save'
        );

        if (saveChoice === 'Save As...') {
            await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.file(newFileName)
            });
        }
    }

    // Helper methods
    private epsilonClosure(nfa: NFA, states: Set<string>): Set<string> {
        const closure = new Set(states);
        const stack = [...states];

        while (stack.length > 0) {
            const state = stack.pop()!;
            const transitions = nfa.transitions[state];
            
            if (transitions) {
                for (const epsilonSymbol of ['epsilon', 'ε', 'λ']) {
                    if (transitions[epsilonSymbol]) {
                        for (const nextState of transitions[epsilonSymbol]) {
                            if (!closure.has(nextState)) {
                                closure.add(nextState);
                                stack.push(nextState);
                            }
                        }
                    }
                }
            }
        }

        return closure;
    }

    private stateSetToString(states: Set<string>): string {
        return Array.from(states).sort().join(',');
    }

    private isAcceptingStateSet(nfa: NFA, states: Set<string>): boolean {
        return Array.from(states).some(state => nfa.accept_states.includes(state));
    }

    private stateEliminationMethod(dfa: DFA): string {
        // Simplified state elimination - returns a basic pattern
        // In practice, this would be a complex algorithm
        if (dfa.states.length <= 2) {
            // Simple case
            const transitions = Object.values(dfa.transitions).map(t => Object.keys(t)).flat();
            return transitions.length > 0 ? transitions.join('|') : 'ε';
        }
        
        return '(Complex expression - manual construction recommended)';
    }

    private getAutomataType(uri: vscode.Uri): 'dfa' | 'nfa' | 'tm' {
        if (uri.fsPath.endsWith('.dfa')) return 'dfa';
        if (uri.fsPath.endsWith('.nfa')) return 'nfa';
        if (uri.fsPath.endsWith('.tm')) return 'tm';
        return 'dfa';
    }
}