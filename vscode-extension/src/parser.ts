export interface AutomatonBase {
    states: string[];
    alphabet: string[];
    start_state: string;
    accept_states: string[];
    transitions: { [state: string]: { [symbol: string]: any } };
}

export interface DFA extends AutomatonBase {
    type: 'dfa';
    transitions: { [state: string]: { [symbol: string]: string } };
}

export interface NFA extends AutomatonBase {
    type: 'nfa';
    transitions: { [state: string]: { [symbol: string]: string[] } };
}

export interface TuringMachine extends AutomatonBase {
    type: 'tm';
    tape_alphabet: string[];
    blank_symbol: string;
    reject_states?: string[];
    transitions: { 
        [state: string]: { 
            [symbol: string]: {
                state: string;
                write: string;
                move: 'L' | 'R' | 'S';
            }
        } 
    };
}

export type Automaton = DFA | NFA | TuringMachine;

export class AutomataParser {
    parse(text: string, type: 'dfa' | 'nfa' | 'tm'): Automaton {
        try {
            const json = JSON.parse(text);
            return this.validateAndConvert(json, type);
        } catch (error) {
            throw new Error(`JSON parse error: ${error instanceof Error ? error.message : 'Invalid JSON'}`);
        }
    }

    private validateAndConvert(json: any, type: 'dfa' | 'nfa' | 'tm'): Automaton {
        // Validate required fields
        this.validateRequiredFields(json, type);
        
        // Validate field types
        this.validateFieldTypes(json, type);
        
        // Validate semantic constraints
        this.validateSemanticConstraints(json, type);
        
        // Convert to proper automaton type
        return { ...json, type } as Automaton;
    }

    private validateRequiredFields(json: any, type: 'dfa' | 'nfa' | 'tm'): void {
        const requiredFields = ['states', 'alphabet', 'start_state', 'accept_states', 'transitions'];
        
        if (type === 'tm') {
            requiredFields.push('tape_alphabet', 'blank_symbol');
        }

        for (const field of requiredFields) {
            if (!(field in json)) {
                throw new Error(`Missing required field: ${field}`);
            }
        }
    }

    private validateFieldTypes(json: any, type: 'dfa' | 'nfa' | 'tm'): void {
        // Validate arrays
        if (!Array.isArray(json.states)) {
            throw new Error('states must be an array');
        }
        if (!Array.isArray(json.alphabet)) {
            throw new Error('alphabet must be an array');
        }
        if (!Array.isArray(json.accept_states)) {
            throw new Error('accept_states must be an array');
        }

        if (type === 'tm') {
            if (!Array.isArray(json.tape_alphabet)) {
                throw new Error('tape_alphabet must be an array');
            }
            if (typeof json.blank_symbol !== 'string') {
                throw new Error('blank_symbol must be a string');
            }
            if (json.reject_states && !Array.isArray(json.reject_states)) {
                throw new Error('reject_states must be an array');
            }
        }

        // Validate start_state
        if (typeof json.start_state !== 'string') {
            throw new Error('start_state must be a string');
        }

        // Validate transitions
        if (typeof json.transitions !== 'object' || json.transitions === null) {
            throw new Error('transitions must be an object');
        }

        // Validate transition structure based on type
        this.validateTransitionStructure(json.transitions, type);
    }

    private validateTransitionStructure(transitions: any, type: 'dfa' | 'nfa' | 'tm'): void {
        for (const [state, stateTransitions] of Object.entries(transitions)) {
            if (typeof stateTransitions !== 'object' || stateTransitions === null) {
                throw new Error(`Transitions for state ${state} must be an object`);
            }

            for (const [symbol, destination] of Object.entries(stateTransitions as any)) {
                switch (type) {
                    case 'dfa':
                        if (typeof destination !== 'string') {
                            throw new Error(`DFA transition ${state} -> ${symbol} must be a string`);
                        }
                        break;
                    case 'nfa':
                        if (!Array.isArray(destination)) {
                            throw new Error(`NFA transition ${state} -> ${symbol} must be an array`);
                        }
                        for (const dest of destination) {
                            if (typeof dest !== 'string') {
                                throw new Error(`NFA transition destinations must be strings`);
                            }
                        }
                        break;
                    case 'tm':
                        if (typeof destination !== 'object' || destination === null) {
                            throw new Error(`TM transition ${state} -> ${symbol} must be an object`);
                        }
                        const tmDest = destination as any;
                        if (typeof tmDest.state !== 'string') {
                            throw new Error(`TM transition must have a string 'state' field`);
                        }
                        if (typeof tmDest.write !== 'string') {
                            throw new Error(`TM transition must have a string 'write' field`);
                        }
                        if (!['L', 'R', 'S'].includes(tmDest.move)) {
                            throw new Error(`TM transition 'move' must be 'L', 'R', or 'S'`);
                        }
                        break;
                }
            }
        }
    }

    private validateSemanticConstraints(json: any, type: 'dfa' | 'nfa' | 'tm'): void {
        const states = new Set(json.states);
        const alphabet = new Set(json.alphabet);

        // Validate start_state is in states
        if (!states.has(json.start_state)) {
            throw new Error(`start_state '${json.start_state}' is not in states`);
        }

        // Validate accept_states are in states
        for (const acceptState of json.accept_states) {
            if (!states.has(acceptState)) {
                throw new Error(`accept_state '${acceptState}' is not in states`);
            }
        }

        // Validate reject_states for TM
        if (type === 'tm' && json.reject_states) {
            for (const rejectState of json.reject_states) {
                if (!states.has(rejectState)) {
                    throw new Error(`reject_state '${rejectState}' is not in states`);
                }
            }
        }

        // Validate tape_alphabet for TM
        if (type === 'tm') {
            const tapeAlphabet = new Set(json.tape_alphabet);
            
            // Input alphabet must be subset of tape alphabet
            for (const symbol of json.alphabet) {
                if (!tapeAlphabet.has(symbol)) {
                    throw new Error(`Input symbol '${symbol}' not in tape_alphabet`);
                }
            }

            // Blank symbol must be in tape alphabet
            if (!tapeAlphabet.has(json.blank_symbol)) {
                throw new Error(`blank_symbol '${json.blank_symbol}' not in tape_alphabet`);
            }

            // Blank symbol should not be in input alphabet
            if (alphabet.has(json.blank_symbol)) {
                throw new Error(`blank_symbol '${json.blank_symbol}' should not be in input alphabet`);
            }
        }

        // Validate transitions reference valid states and symbols
        this.validateTransitionReferences(json.transitions, states, alphabet, type, json.tape_alphabet);
    }

    private validateTransitionReferences(
        transitions: any, 
        states: Set<string>, 
        alphabet: Set<string>, 
        type: 'dfa' | 'nfa' | 'tm',
        tapeAlphabet?: string[]
    ): void {
        const validSymbols = type === 'tm' ? new Set(tapeAlphabet) : alphabet;
        
        // Add epsilon for NFA
        if (type === 'nfa') {
            validSymbols.add('epsilon');
            validSymbols.add('ε');
            validSymbols.add('λ');
        }

        for (const [state, stateTransitions] of Object.entries(transitions)) {
            if (!states.has(state)) {
                throw new Error(`Transition state '${state}' is not in states`);
            }

            for (const [symbol, destination] of Object.entries(stateTransitions as any)) {
                if (!validSymbols.has(symbol)) {
                    throw new Error(`Transition symbol '${symbol}' from state '${state}' is not in alphabet`);
                }

                // Validate destination states
                switch (type) {
                    case 'dfa':
                        if (!states.has(destination as string)) {
                            throw new Error(`Transition destination '${destination}' is not in states`);
                        }
                        break;
                    case 'nfa':
                        for (const dest of destination as string[]) {
                            if (!states.has(dest)) {
                                throw new Error(`Transition destination '${dest}' is not in states`);
                            }
                        }
                        break;
                    case 'tm':
                        const tmDest = destination as any;
                        if (!states.has(tmDest.state)) {
                            throw new Error(`TM transition destination state '${tmDest.state}' is not in states`);
                        }
                        if (!validSymbols.has(tmDest.write)) {
                            throw new Error(`TM transition write symbol '${tmDest.write}' is not in tape_alphabet`);
                        }
                        break;
                }
            }
        }
    }

    // Helper method to extract all referenced states from automaton
    getReferencedStates(automaton: Automaton): Set<string> {
        const referenced = new Set<string>();
        referenced.add(automaton.start_state);
        
        for (const state of automaton.accept_states) {
            referenced.add(state);
        }

        if (automaton.type === 'tm' && automaton.reject_states) {
            for (const state of automaton.reject_states) {
                referenced.add(state);
            }
        }

        for (const [state, stateTransitions] of Object.entries(automaton.transitions)) {
            referenced.add(state);
            
            for (const destination of Object.values(stateTransitions)) {
                switch (automaton.type) {
                    case 'dfa':
                        referenced.add(destination as string);
                        break;
                    case 'nfa':
                        for (const dest of destination as string[]) {
                            referenced.add(dest);
                        }
                        break;
                    case 'tm':
                        referenced.add((destination as any).state);
                        break;
                }
            }
        }

        return referenced;
    }

    // Helper method to check if automaton is deterministic
    isDeterministic(automaton: NFA): boolean {
        if (automaton.type !== 'nfa') return true;

        for (const stateTransitions of Object.values(automaton.transitions)) {
            for (const [symbol, destinations] of Object.entries(stateTransitions)) {
                // Check for epsilon transitions
                if (symbol === 'epsilon' || symbol === 'ε' || symbol === 'λ') {
                    return false;
                }
                // Check for multiple destinations
                if (destinations.length > 1) {
                    return false;
                }
            }
        }
        return true;
    }
}