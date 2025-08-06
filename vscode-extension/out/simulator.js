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
exports.AutomataSimulator = void 0;
const vscode = __importStar(require("vscode"));
const parser_1 = require("./parser");
class AutomataSimulator {
    constructor() {
        this.parser = new parser_1.AutomataParser();
    }
    async simulate(document) {
        try {
            const text = document.getText();
            const type = this.getAutomataType(document.uri);
            const automaton = this.parser.parse(text, type);
            const input = await vscode.window.showInputBox({
                prompt: 'Enter input string to simulate',
                placeHolder: 'Input string (e.g., "101", "abc", leave empty for empty string)',
                value: ''
            });
            if (input === undefined)
                return; // User cancelled
            await this.runSimulation(automaton, input);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Simulation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    async testInput(document, input) {
        try {
            const text = document.getText();
            const type = this.getAutomataType(document.uri);
            const automaton = this.parser.parse(text, type);
            const result = await this.runSimulation(automaton, input);
            const message = result.accepted
                ? `Input "${input}" is ACCEPTED by the automaton`
                : `Input "${input}" is REJECTED by the automaton`;
            const action = result.accepted ? 'showInformationMessage' : 'showWarningMessage';
            vscode.window[action](message);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Test failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    async runSimulation(automaton, input) {
        switch (automaton.type) {
            case 'dfa':
                return this.simulateDFA(automaton, input);
            case 'nfa':
                return this.simulateNFA(automaton, input);
            case 'tm':
                return this.simulateTuringMachine(automaton, input);
            default:
                throw new Error(`Unknown automaton type: ${automaton.type}`);
        }
    }
    async simulateDFA(dfa, input) {
        const trace = [];
        let currentState = dfa.start_state;
        trace.push({
            step: 0,
            state: currentState,
            input: input,
            position: 0,
            remaining: input
        });
        for (let i = 0; i < input.length; i++) {
            const symbol = input[i];
            const transitions = dfa.transitions[currentState];
            if (!transitions || !transitions[symbol]) {
                trace.push({
                    step: i + 1,
                    state: 'REJECTED',
                    input: input,
                    position: i + 1,
                    remaining: input.substring(i + 1),
                    reason: `No transition from state ${currentState} on symbol ${symbol}`
                });
                return { accepted: false, trace };
            }
            currentState = transitions[symbol];
            trace.push({
                step: i + 1,
                state: currentState,
                input: input,
                position: i + 1,
                remaining: input.substring(i + 1),
                transition: `δ(${trace[trace.length - 1].state}, ${symbol}) = ${currentState}`
            });
        }
        const accepted = dfa.accept_states.includes(currentState);
        trace.push({
            step: input.length + 1,
            state: currentState,
            result: accepted ? 'ACCEPTED' : 'REJECTED',
            reason: accepted
                ? `Final state ${currentState} is accepting`
                : `Final state ${currentState} is not accepting`
        });
        await this.showSimulationTrace('DFA Simulation', trace);
        return { accepted, trace };
    }
    async simulateNFA(nfa, input) {
        const trace = [];
        let currentStates = new Set([nfa.start_state]);
        // Add epsilon closure of start state
        currentStates = this.epsilonClosure(nfa, currentStates);
        trace.push({
            step: 0,
            states: Array.from(currentStates),
            input: input,
            position: 0,
            remaining: input
        });
        for (let i = 0; i < input.length; i++) {
            const symbol = input[i];
            const nextStates = new Set();
            for (const state of currentStates) {
                const transitions = nfa.transitions[state];
                if (transitions && transitions[symbol]) {
                    for (const nextState of transitions[symbol]) {
                        nextStates.add(nextState);
                    }
                }
            }
            if (nextStates.size === 0) {
                trace.push({
                    step: i + 1,
                    states: [],
                    result: 'REJECTED',
                    reason: `No transitions available from states {${Array.from(currentStates).join(', ')}} on symbol ${symbol}`
                });
                return { accepted: false, trace };
            }
            currentStates = this.epsilonClosure(nfa, nextStates);
            trace.push({
                step: i + 1,
                states: Array.from(currentStates),
                input: input,
                position: i + 1,
                remaining: input.substring(i + 1)
            });
        }
        const accepted = Array.from(currentStates).some(state => nfa.accept_states.includes(state));
        trace.push({
            step: input.length + 1,
            states: Array.from(currentStates),
            result: accepted ? 'ACCEPTED' : 'REJECTED',
            reason: accepted
                ? `At least one final state is accepting: ${Array.from(currentStates).filter(s => nfa.accept_states.includes(s)).join(', ')}`
                : `No accepting states in final configuration: {${Array.from(currentStates).join(', ')}}`
        });
        await this.showSimulationTrace('NFA Simulation', trace);
        return { accepted, trace };
    }
    async simulateTuringMachine(tm, input) {
        const trace = [];
        let currentState = tm.start_state;
        let tapePosition = 0;
        let tape = [...input, ...Array(Math.max(10, input.length)).fill(tm.blank_symbol)];
        let step = 0;
        const maxSteps = 1000; // Prevent infinite loops
        trace.push({
            step: step++,
            state: currentState,
            tape: [...tape],
            position: tapePosition,
            tapeView: this.formatTape(tape, tapePosition)
        });
        while (step < maxSteps) {
            // Check if in accepting or rejecting state
            if (tm.accept_states.includes(currentState)) {
                trace.push({
                    step: step,
                    state: currentState,
                    result: 'ACCEPTED',
                    reason: `Reached accepting state ${currentState}`
                });
                await this.showTuringMachineTrace('Turing Machine Simulation', trace);
                return { accepted: true, trace };
            }
            if (tm.reject_states && tm.reject_states.includes(currentState)) {
                trace.push({
                    step: step,
                    state: currentState,
                    result: 'REJECTED',
                    reason: `Reached rejecting state ${currentState}`
                });
                await this.showTuringMachineTrace('Turing Machine Simulation', trace);
                return { accepted: false, trace };
            }
            const currentSymbol = tape[tapePosition];
            const transitions = tm.transitions[currentState];
            if (!transitions || !transitions[currentSymbol]) {
                trace.push({
                    step: step,
                    state: currentState,
                    result: 'REJECTED',
                    reason: `No transition from state ${currentState} on symbol '${currentSymbol}'`
                });
                await this.showTuringMachineTrace('Turing Machine Simulation', trace);
                return { accepted: false, trace };
            }
            const transition = transitions[currentSymbol];
            // Write to tape
            tape[tapePosition] = transition.write;
            // Move tape head
            if (transition.move === 'L') {
                tapePosition = Math.max(0, tapePosition - 1);
            }
            else if (transition.move === 'R') {
                tapePosition++;
                // Extend tape if necessary
                if (tapePosition >= tape.length) {
                    tape.push(...Array(10).fill(tm.blank_symbol));
                }
            }
            // 'S' means stay
            currentState = transition.state;
            trace.push({
                step: step++,
                state: currentState,
                tape: [...tape],
                position: tapePosition,
                tapeView: this.formatTape(tape, tapePosition),
                transition: `δ(${trace[trace.length - 1].state}, ${currentSymbol}) = (${currentState}, ${transition.write}, ${transition.move})`
            });
        }
        trace.push({
            step: step,
            state: currentState,
            result: 'TIMEOUT',
            reason: `Simulation stopped after ${maxSteps} steps to prevent infinite loop`
        });
        await this.showTuringMachineTrace('Turing Machine Simulation', trace);
        return { accepted: false, trace };
    }
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
    formatTape(tape, position, windowSize = 10) {
        const start = Math.max(0, position - windowSize / 2);
        const end = Math.min(tape.length, position + windowSize / 2 + 1);
        let result = '';
        for (let i = start; i < end; i++) {
            if (i === position) {
                result += `[${tape[i]}]`;
            }
            else {
                result += ` ${tape[i]} `;
            }
        }
        return result;
    }
    async showSimulationTrace(title, trace) {
        const options = ['Show in Output', 'Copy to Clipboard', 'Save to File'];
        const choice = await vscode.window.showInformationMessage(`${title} completed. View trace?`, ...options);
        if (!choice)
            return;
        const traceText = this.formatTrace(trace);
        switch (choice) {
            case 'Show in Output':
                this.showInOutput(title, traceText);
                break;
            case 'Copy to Clipboard':
                await vscode.env.clipboard.writeText(traceText);
                vscode.window.showInformationMessage('Trace copied to clipboard');
                break;
            case 'Save to File':
                await this.saveTraceToFile(title, traceText);
                break;
        }
    }
    async showTuringMachineTrace(title, trace) {
        const options = ['Show in Output', 'Copy to Clipboard', 'Save to File'];
        const choice = await vscode.window.showInformationMessage(`${title} completed. View trace?`, ...options);
        if (!choice)
            return;
        const traceText = this.formatTuringMachineTrace(trace);
        switch (choice) {
            case 'Show in Output':
                this.showInOutput(title, traceText);
                break;
            case 'Copy to Clipboard':
                await vscode.env.clipboard.writeText(traceText);
                vscode.window.showInformationMessage('Trace copied to clipboard');
                break;
            case 'Save to File':
                await this.saveTraceToFile(title, traceText);
                break;
        }
    }
    formatTrace(trace) {
        let result = 'SIMULATION TRACE\n';
        result += '=================\n\n';
        for (const step of trace) {
            result += `Step ${step.step}:\n`;
            if (step.states) {
                result += `  States: {${step.states.join(', ')}}\n`;
            }
            else if (step.state) {
                result += `  State: ${step.state}\n`;
            }
            if (step.input !== undefined && step.position !== undefined) {
                result += `  Input: "${step.input}" (position: ${step.position})\n`;
                result += `  Remaining: "${step.remaining || ''}"\n`;
            }
            if (step.transition) {
                result += `  Transition: ${step.transition}\n`;
            }
            if (step.reason) {
                result += `  Reason: ${step.reason}\n`;
            }
            if (step.result) {
                result += `  RESULT: ${step.result}\n`;
            }
            result += '\n';
        }
        return result;
    }
    formatTuringMachineTrace(trace) {
        let result = 'TURING MACHINE SIMULATION TRACE\n';
        result += '================================\n\n';
        for (const step of trace) {
            result += `Step ${step.step}:\n`;
            result += `  State: ${step.state}\n`;
            if (step.tapeView) {
                result += `  Tape: ${step.tapeView}\n`;
                result += `  Head Position: ${step.position}\n`;
            }
            if (step.transition) {
                result += `  Transition: ${step.transition}\n`;
            }
            if (step.reason) {
                result += `  Reason: ${step.reason}\n`;
            }
            if (step.result) {
                result += `  RESULT: ${step.result}\n`;
            }
            result += '\n';
        }
        return result;
    }
    showInOutput(title, content) {
        const outputChannel = vscode.window.createOutputChannel(title);
        outputChannel.clear();
        outputChannel.append(content);
        outputChannel.show();
    }
    async saveTraceToFile(title, content) {
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(`${title.toLowerCase().replace(/\s+/g, '_')}_trace.txt`),
            filters: {
                'Text files': ['txt'],
                'All files': ['*']
            }
        });
        if (uri) {
            await vscode.workspace.fs.writeFile(uri, Buffer.from(content));
            vscode.window.showInformationMessage(`Trace saved to ${uri.fsPath}`);
        }
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
exports.AutomataSimulator = AutomataSimulator;
//# sourceMappingURL=simulator.js.map