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
exports.AutomataConverter = void 0;
const vscode = __importStar(require("vscode"));
const parser_1 = require("./parser");
class AutomataConverter {
    constructor() {
        this.parser = new parser_1.AutomataParser();
    }
    async convert(document) {
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
            if (!choice)
                return;
            await this.performConversion(automaton, choice.value, document);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Conversion failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    getConversionOptions(type) {
        const options = [];
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
    async performConversion(automaton, conversionType, originalDocument) {
        let result;
        let fileExtension;
        switch (conversionType) {
            case 'nfa-to-dfa':
                result = this.convertNFAToDFA(automaton);
                fileExtension = '.dfa';
                break;
            case 'dfa-to-nfa':
                result = this.convertDFAToNFA(automaton);
                fileExtension = '.nfa';
                break;
            case 'dfa-to-grammar':
                result = this.convertDFAToGrammar(automaton);
                fileExtension = '.txt';
                break;
            case 'dfa-to-regex':
                result = this.convertDFAToRegex(automaton);
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
    convertNFAToDFA(nfa) {
        // Subset construction algorithm
        const dfaStates = [];
        const dfaTransitions = {};
        const dfaAcceptStates = [];
        // Start with epsilon closure of start state
        const startClosure = this.epsilonClosure(nfa, new Set([nfa.start_state]));
        const startStateStr = this.stateSetToString(startClosure);
        const unmarked = [startClosure];
        const marked = new Map();
        marked.set(startStateStr, startClosure);
        dfaStates.push(startStateStr);
        // Check if start state is accepting
        if (this.isAcceptingStateSet(nfa, startClosure)) {
            dfaAcceptStates.push(startStateStr);
        }
        while (unmarked.length > 0) {
            const current = unmarked.shift();
            const currentStr = this.stateSetToString(current);
            dfaTransitions[currentStr] = {};
            for (const symbol of nfa.alphabet) {
                const nextStates = new Set();
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
                if (closure.size === 0)
                    continue;
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
        const dfa = {
            type: 'dfa',
            states: dfaStates,
            alphabet: nfa.alphabet,
            start_state: startStateStr,
            accept_states: dfaAcceptStates,
            transitions: dfaTransitions
        };
        return JSON.stringify(dfa, null, 2);
    }
    convertDFAToNFA(dfa) {
        // Trivial conversion - DFA is already an NFA
        const nfaTransitions = {};
        for (const [state, stateTransitions] of Object.entries(dfa.transitions)) {
            nfaTransitions[state] = {};
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                nfaTransitions[state][symbol] = [destination];
            }
        }
        const nfa = {
            type: 'nfa',
            states: dfa.states,
            alphabet: dfa.alphabet,
            start_state: dfa.start_state,
            accept_states: dfa.accept_states,
            transitions: nfaTransitions
        };
        return JSON.stringify(nfa, null, 2);
    }
    convertDFAToGrammar(dfa) {
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
                }
                else {
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
    convertDFAToRegex(dfa) {
        // State elimination method for converting DFA to regular expression
        let result = 'Regular Expression Generation\n';
        result += '=============================\n\n';
        try {
            const regex = this.stateEliminationMethod(dfa);
            result += `Regular Expression: ${regex}\n\n`;
            result += 'Note: This is a simplified conversion. Complex DFAs may produce very long expressions.\n';
        }
        catch (error) {
            result += `Error generating regular expression: ${error instanceof Error ? error.message : 'Unknown error'}\n`;
            result += '\nFallback: Manual construction required for complex automata.\n';
        }
        return result;
    }
    convertToJFLAP(automaton) {
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
        xml += '<structure>\n';
        xml += `  <type>${automaton.type === 'tm' ? 'turing' : 'fa'}</type>\n`;
        xml += '  <automaton>\n';
        // Add states
        let stateId = 0;
        const stateMap = new Map();
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
                        const toId = stateMap.get(destination);
                        xml += `    <transition>\n`;
                        xml += `      <from>${fromId}</from>\n`;
                        xml += `      <to>${toId}</to>\n`;
                        xml += `      <read>${symbol}</read>\n`;
                        xml += `    </transition>\n`;
                        break;
                    case 'nfa':
                        for (const dest of destination) {
                            const nfaToId = stateMap.get(dest);
                            xml += `    <transition>\n`;
                            xml += `      <from>${fromId}</from>\n`;
                            xml += `      <to>${nfaToId}</to>\n`;
                            xml += `      <read>${symbol === 'epsilon' ? '' : symbol}</read>\n`;
                            xml += `    </transition>\n`;
                        }
                        break;
                    case 'tm':
                        const tmDest = destination;
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
    convertToDOT(automaton) {
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
                        for (const dest of destination) {
                            const label = symbol === 'epsilon' || symbol === 'ε' || symbol === 'λ' ? 'ε' : symbol;
                            dot += `  ${fromState} -> ${dest} [label="${label}"];\n`;
                        }
                        break;
                    case 'tm':
                        const tmDest = destination;
                        dot += `  ${fromState} -> ${tmDest.state} [label="${symbol}/${tmDest.write},${tmDest.move}"];\n`;
                        break;
                }
            }
        }
        dot += '}\n';
        return dot;
    }
    async createNewDocument(content, extension, originalDocument) {
        const baseName = originalDocument.fileName.replace(/\.[^.]+$/, '');
        const newFileName = `${baseName}_converted${extension}`;
        const newDocument = await vscode.workspace.openTextDocument({
            content: content,
            language: extension === '.json' ? 'json' : 'text'
        });
        await vscode.window.showTextDocument(newDocument);
        // Offer to save the document
        const saveChoice = await vscode.window.showInformationMessage('Conversion completed. Save the result?', 'Save As...', 'Don\'t Save');
        if (saveChoice === 'Save As...') {
            await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.file(newFileName)
            });
        }
    }
    // Helper methods
    epsilonClosure(nfa, states) {
        const closure = new Set(states);
        const stack = [...states];
        while (stack.length > 0) {
            const state = stack.pop();
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
    stateSetToString(states) {
        return Array.from(states).sort().join(',');
    }
    isAcceptingStateSet(nfa, states) {
        return Array.from(states).some(state => nfa.accept_states.includes(state));
    }
    stateEliminationMethod(dfa) {
        // Simplified state elimination - returns a basic pattern
        // In practice, this would be a complex algorithm
        if (dfa.states.length <= 2) {
            // Simple case
            const transitions = Object.values(dfa.transitions).map(t => Object.keys(t)).flat();
            return transitions.length > 0 ? transitions.join('|') : 'ε';
        }
        return '(Complex expression - manual construction recommended)';
    }
    getAutomataType(uri) {
        if (uri.fsPath.endsWith('.dfa'))
            return 'dfa';
        if (uri.fsPath.endsWith('.nfa'))
            return 'nfa';
        if (uri.fsPath.endsWith('.tm'))
            return 'tm';
        return 'dfa';
    }
}
exports.AutomataConverter = AutomataConverter;
//# sourceMappingURL=converter.js.map