"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AutomataValidator = void 0;
const vscode_languageserver_types_1 = require("vscode-languageserver-types");
class AutomataValidator {
    validate(automaton, document) {
        const diagnostics = [];
        // Basic structural validation
        this.validateStructure(automaton, document, diagnostics);
        // Type-specific validation
        switch (automaton.type) {
            case 'dfa':
                this.validateDFA(automaton, document, diagnostics);
                break;
            case 'nfa':
                this.validateNFA(automaton, document, diagnostics);
                break;
            case 'tm':
                this.validateTM(automaton, document, diagnostics);
                break;
        }
        // Common semantic validation
        this.validateSemantics(automaton, document, diagnostics);
        return diagnostics;
    }
    validateStructure(automaton, document, diagnostics) {
        // Check for empty sets
        if (automaton.states.length === 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                range: this.findFieldRange(document, 'states') || this.getDefaultRange(),
                message: 'States set cannot be empty',
                source: 'automata-validator'
            });
        }
        if (automaton.alphabet.length === 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                range: this.findFieldRange(document, 'alphabet') || this.getDefaultRange(),
                message: 'Alphabet cannot be empty',
                source: 'automata-validator'
            });
        }
        // Check for duplicates
        const duplicateStates = this.findDuplicates(automaton.states);
        if (duplicateStates.length > 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                range: this.findFieldRange(document, 'states') || this.getDefaultRange(),
                message: `Duplicate states found: ${duplicateStates.join(', ')}`,
                source: 'automata-validator'
            });
        }
        const duplicateSymbols = this.findDuplicates(automaton.alphabet);
        if (duplicateSymbols.length > 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                range: this.findFieldRange(document, 'alphabet') || this.getDefaultRange(),
                message: `Duplicate alphabet symbols found: ${duplicateSymbols.join(', ')}`,
                source: 'automata-validator'
            });
        }
        // Check for unreachable states
        const reachableStates = this.getReachableStates(automaton);
        const unreachableStates = automaton.states.filter(state => !reachableStates.has(state));
        if (unreachableStates.length > 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Warning,
                range: this.findFieldRange(document, 'states') || this.getDefaultRange(),
                message: `Unreachable states found: ${unreachableStates.join(', ')}`,
                source: 'automata-validator'
            });
        }
        // Check for undefined transitions
        const missingTransitions = this.findMissingTransitions(automaton);
        if (missingTransitions.length > 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Warning,
                range: this.findFieldRange(document, 'transitions') || this.getDefaultRange(),
                message: `Missing transitions: ${missingTransitions.join(', ')}`,
                source: 'automata-validator'
            });
        }
    }
    validateDFA(dfa, document, diagnostics) {
        // Check completeness - every state should have transitions for every alphabet symbol
        for (const state of dfa.states) {
            const stateTransitions = dfa.transitions[state] || {};
            for (const symbol of dfa.alphabet) {
                if (!(symbol in stateTransitions)) {
                    diagnostics.push({
                        severity: vscode_languageserver_types_1.DiagnosticSeverity.Warning,
                        range: this.findTransitionRange(document, state, symbol) || this.getDefaultRange(),
                        message: `DFA missing transition from state '${state}' on symbol '${symbol}'`,
                        source: 'automata-validator'
                    });
                }
            }
        }
        // Check for trap states (states with no outgoing transitions to accepting states)
        const trapStates = this.findTrapStates(dfa);
        if (trapStates.length > 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Information,
                range: this.findFieldRange(document, 'states') || this.getDefaultRange(),
                message: `Potential trap states found: ${trapStates.join(', ')} (no path to accepting states)`,
                source: 'automata-validator'
            });
        }
    }
    validateNFA(nfa, document, diagnostics) {
        // Check for epsilon transitions
        let hasEpsilonTransitions = false;
        for (const [state, stateTransitions] of Object.entries(nfa.transitions)) {
            for (const symbol of Object.keys(stateTransitions)) {
                if (symbol === 'epsilon' || symbol === 'ε' || symbol === 'λ') {
                    hasEpsilonTransitions = true;
                    break;
                }
            }
            if (hasEpsilonTransitions)
                break;
        }
        if (hasEpsilonTransitions) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Information,
                range: this.findFieldRange(document, 'transitions') || this.getDefaultRange(),
                message: 'NFA contains epsilon transitions',
                source: 'automata-validator'
            });
        }
        // Check if NFA is actually deterministic
        const isDeterministic = this.isNFADeterministic(nfa);
        if (isDeterministic && !hasEpsilonTransitions) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Information,
                range: this.getDefaultRange(),
                message: 'This NFA is actually deterministic and could be converted to a DFA',
                source: 'automata-validator'
            });
        }
    }
    validateTM(tm, document, diagnostics) {
        // Check that tape alphabet is a superset of input alphabet
        const inputSymbols = new Set(tm.alphabet);
        const tapeSymbols = new Set(tm.tape_alphabet);
        for (const symbol of tm.alphabet) {
            if (!tapeSymbols.has(symbol)) {
                diagnostics.push({
                    severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                    range: this.findFieldRange(document, 'tape_alphabet') || this.getDefaultRange(),
                    message: `Input symbol '${symbol}' not found in tape alphabet`,
                    source: 'automata-validator'
                });
            }
        }
        // Check blank symbol
        if (!tapeSymbols.has(tm.blank_symbol)) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                range: this.findFieldRange(document, 'blank_symbol') || this.getDefaultRange(),
                message: `Blank symbol '${tm.blank_symbol}' not found in tape alphabet`,
                source: 'automata-validator'
            });
        }
        if (inputSymbols.has(tm.blank_symbol)) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Warning,
                range: this.findFieldRange(document, 'blank_symbol') || this.getDefaultRange(),
                message: `Blank symbol '${tm.blank_symbol}' should not be in input alphabet`,
                source: 'automata-validator'
            });
        }
        // Check for conflicting accept/reject states
        if (tm.reject_states) {
            const acceptSet = new Set(tm.accept_states);
            const rejectSet = new Set(tm.reject_states);
            const overlap = tm.accept_states.filter(state => rejectSet.has(state));
            if (overlap.length > 0) {
                diagnostics.push({
                    severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                    range: this.findFieldRange(document, 'accept_states') || this.getDefaultRange(),
                    message: `States cannot be both accepting and rejecting: ${overlap.join(', ')}`,
                    source: 'automata-validator'
                });
            }
        }
        // Check for infinite loops (states with transitions to themselves that don't change the tape)
        const potentialInfiniteLoops = this.findPotentialInfiniteLoops(tm);
        if (potentialInfiniteLoops.length > 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Warning,
                range: this.findFieldRange(document, 'transitions') || this.getDefaultRange(),
                message: `Potential infinite loops detected in states: ${potentialInfiniteLoops.join(', ')}`,
                source: 'automata-validator'
            });
        }
    }
    validateSemantics(automaton, document, diagnostics) {
        // Check if start state can reach any accept state
        const canReachAccept = this.canReachAcceptingState(automaton);
        if (!canReachAccept) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Warning,
                range: this.findFieldRange(document, 'start_state') || this.getDefaultRange(),
                message: 'Start state cannot reach any accepting state',
                source: 'automata-validator'
            });
        }
        // Check for dead states (states that cannot reach accepting states)
        const deadStates = this.findDeadStates(automaton);
        if (deadStates.length > 0) {
            diagnostics.push({
                severity: vscode_languageserver_types_1.DiagnosticSeverity.Information,
                range: this.findFieldRange(document, 'states') || this.getDefaultRange(),
                message: `Dead states found (cannot reach accepting states): ${deadStates.join(', ')}`,
                source: 'automata-validator'
            });
        }
    }
    // Helper methods
    findDuplicates(array) {
        const seen = new Set();
        const duplicates = new Set();
        for (const item of array) {
            if (seen.has(item)) {
                duplicates.add(item);
            }
            else {
                seen.add(item);
            }
        }
        return Array.from(duplicates);
    }
    getReachableStates(automaton) {
        const reachable = new Set();
        const queue = [automaton.start_state];
        while (queue.length > 0) {
            const current = queue.shift();
            if (reachable.has(current))
                continue;
            reachable.add(current);
            const stateTransitions = automaton.transitions[current];
            if (stateTransitions) {
                for (const destinations of Object.values(stateTransitions)) {
                    switch (automaton.type) {
                        case 'dfa':
                            if (!reachable.has(destinations)) {
                                queue.push(destinations);
                            }
                            break;
                        case 'nfa':
                            for (const dest of destinations) {
                                if (!reachable.has(dest)) {
                                    queue.push(dest);
                                }
                            }
                            break;
                        case 'tm':
                            const tmDest = destinations.state;
                            if (!reachable.has(tmDest)) {
                                queue.push(tmDest);
                            }
                            break;
                    }
                }
            }
        }
        return reachable;
    }
    findMissingTransitions(automaton) {
        const missing = [];
        for (const state of automaton.states) {
            const stateTransitions = automaton.transitions[state];
            if (!stateTransitions) {
                missing.push(`${state}: all transitions missing`);
                continue;
            }
            const alphabet = automaton.type === 'tm' ?
                automaton.tape_alphabet :
                automaton.alphabet;
            for (const symbol of alphabet) {
                if (!(symbol in stateTransitions)) {
                    missing.push(`${state} on ${symbol}`);
                }
            }
        }
        return missing;
    }
    findTrapStates(dfa) {
        const trapStates = [];
        const acceptingStates = new Set(dfa.accept_states);
        for (const state of dfa.states) {
            if (acceptingStates.has(state))
                continue;
            if (!this.canReachAcceptingStateFrom(dfa, state)) {
                trapStates.push(state);
            }
        }
        return trapStates;
    }
    canReachAcceptingStateFrom(automaton, startState) {
        const visited = new Set();
        const queue = [startState];
        const acceptingStates = new Set(automaton.accept_states);
        while (queue.length > 0) {
            const current = queue.shift();
            if (visited.has(current))
                continue;
            visited.add(current);
            if (acceptingStates.has(current))
                return true;
            const stateTransitions = automaton.transitions[current];
            if (stateTransitions) {
                for (const destinations of Object.values(stateTransitions)) {
                    switch (automaton.type) {
                        case 'dfa':
                            queue.push(destinations);
                            break;
                        case 'nfa':
                            queue.push(...destinations);
                            break;
                        case 'tm':
                            queue.push(destinations.state);
                            break;
                    }
                }
            }
        }
        return false;
    }
    canReachAcceptingState(automaton) {
        return this.canReachAcceptingStateFrom(automaton, automaton.start_state);
    }
    findDeadStates(automaton) {
        const deadStates = [];
        for (const state of automaton.states) {
            if (!this.canReachAcceptingStateFrom(automaton, state)) {
                deadStates.push(state);
            }
        }
        return deadStates;
    }
    isNFADeterministic(nfa) {
        for (const stateTransitions of Object.values(nfa.transitions)) {
            for (const [symbol, destinations] of Object.entries(stateTransitions)) {
                if (symbol === 'epsilon' || symbol === 'ε' || symbol === 'λ') {
                    return false;
                }
                if (destinations.length > 1) {
                    return false;
                }
            }
        }
        return true;
    }
    findPotentialInfiniteLoops(tm) {
        const potentialLoops = [];
        for (const [state, stateTransitions] of Object.entries(tm.transitions)) {
            for (const [symbol, transition] of Object.entries(stateTransitions)) {
                if (transition.state === state &&
                    transition.write === symbol &&
                    transition.move === 'S') {
                    potentialLoops.push(state);
                }
            }
        }
        return potentialLoops;
    }
    // Range finding helpers (simplified - in a real implementation, these would parse the document)
    findFieldRange(document, field) {
        const text = document.getText();
        const regex = new RegExp(`"${field}"\\s*:`);
        const match = regex.exec(text);
        if (match) {
            const start = document.positionAt(match.index);
            const end = document.positionAt(match.index + match[0].length);
            return { start, end };
        }
        return null;
    }
    findTransitionRange(document, state, symbol) {
        // Simplified - would need more sophisticated parsing in real implementation
        return null;
    }
    getDefaultRange() {
        return {
            start: { line: 0, character: 0 },
            end: { line: 0, character: 0 }
        };
    }
    async validateDocument(document) {
        try {
            const text = document.getText();
            const json = JSON.parse(text);
            // Determine automaton type from file extension
            let type = 'dfa';
            if (document.uri.endsWith('.nfa'))
                type = 'nfa';
            else if (document.uri.endsWith('.tm'))
                type = 'tm';
            const automaton = { ...json, type };
            return this.validate(automaton, document);
        }
        catch (error) {
            return [{
                    severity: vscode_languageserver_types_1.DiagnosticSeverity.Error,
                    range: this.getDefaultRange(),
                    message: `Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                    source: 'automata-validator'
                }];
        }
    }
}
exports.AutomataValidator = AutomataValidator;
//# sourceMappingURL=validator.js.map