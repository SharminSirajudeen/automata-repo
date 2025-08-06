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
exports.AutomataExporter = void 0;
const vscode = __importStar(require("vscode"));
const parser_1 = require("./parser");
class AutomataExporter {
    constructor() {
        this.parser = new parser_1.AutomataParser();
    }
    async export(document) {
        try {
            const text = document.getText();
            const type = this.getAutomataType(document.uri);
            const automaton = this.parser.parse(text, type);
            const config = vscode.workspace.getConfiguration('automata');
            const defaultFormat = config.get('export.format', 'svg');
            const format = await vscode.window.showQuickPick([
                { label: 'SVG', description: 'Scalable Vector Graphics', value: 'svg' },
                { label: 'PNG', description: 'Portable Network Graphics', value: 'png' },
                { label: 'JSON', description: 'JSON format', value: 'json' },
                { label: 'DOT', description: 'GraphViz DOT format', value: 'dot' },
                { label: 'JFLAP XML', description: 'JFLAP XML format', value: 'jflap' },
                { label: 'LaTeX', description: 'LaTeX tikz format', value: 'latex' },
                { label: 'PDF Report', description: 'Comprehensive PDF report', value: 'pdf' }
            ], {
                placeHolder: 'Select export format',
                selectedItem: { label: defaultFormat.toUpperCase(), value: defaultFormat }
            });
            if (!format)
                return;
            await this.performExport(automaton, format.value, document);
        }
        catch (error) {
            vscode.window.showErrorMessage(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }
    async performExport(automaton, format, document) {
        const baseName = document.fileName.replace(/\.[^.]+$/, '');
        let content;
        let fileExtension;
        let defaultFileName;
        switch (format) {
            case 'svg':
                content = this.generateSVG(automaton);
                fileExtension = '.svg';
                defaultFileName = `${baseName}_diagram.svg`;
                break;
            case 'png':
                await this.exportToPNG(automaton, baseName);
                return;
            case 'json':
                content = JSON.stringify(automaton, null, 2);
                fileExtension = '.json';
                defaultFileName = `${baseName}_export.json`;
                break;
            case 'dot':
                content = this.generateDOT(automaton);
                fileExtension = '.dot';
                defaultFileName = `${baseName}_graph.dot`;
                break;
            case 'jflap':
                content = this.generateJFLAP(automaton);
                fileExtension = '.jff';
                defaultFileName = `${baseName}_jflap.jff`;
                break;
            case 'latex':
                content = this.generateLaTeX(automaton);
                fileExtension = '.tex';
                defaultFileName = `${baseName}_diagram.tex`;
                break;
            case 'pdf':
                await this.exportToPDF(automaton, baseName);
                return;
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
        await this.saveToFile(content, defaultFileName, fileExtension);
    }
    generateSVG(automaton) {
        const width = 800;
        const height = 600;
        const stateRadius = 30;
        let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
        svg += `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">\n`;
        svg += `  <defs>\n`;
        svg += `    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">\n`;
        svg += `      <polygon points="0 0, 10 3.5, 0 7" fill="#000" />\n`;
        svg += `    </marker>\n`;
        svg += `  </defs>\n`;
        // Position states in a grid or circle
        const positions = this.calculateStatePositions(automaton.states, width, height, stateRadius);
        // Draw transitions first (so they appear behind states)
        svg += this.generateSVGTransitions(automaton, positions, stateRadius);
        // Draw states
        svg += this.generateSVGStates(automaton, positions, stateRadius);
        svg += `</svg>\n`;
        return svg;
    }
    generateSVGStates(automaton, positions, radius) {
        let svg = '';
        const acceptingStates = new Set(automaton.accept_states);
        for (const state of automaton.states) {
            const pos = positions.get(state);
            const isAccepting = acceptingStates.has(state);
            const isStart = state === automaton.start_state;
            // Draw state circle
            svg += `  <circle cx="${pos.x}" cy="${pos.y}" r="${radius}" fill="lightblue" stroke="black" stroke-width="2"/>\n`;
            // Draw double circle for accepting states
            if (isAccepting) {
                svg += `  <circle cx="${pos.x}" cy="${pos.y}" r="${radius - 5}" fill="none" stroke="black" stroke-width="2"/>\n`;
            }
            // Draw start arrow
            if (isStart) {
                const startX = pos.x - radius - 40;
                svg += `  <line x1="${startX}" y1="${pos.y}" x2="${pos.x - radius}" y2="${pos.y}" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>\n`;
                svg += `  <text x="${startX - 20}" y="${pos.y + 5}" font-family="Arial" font-size="12" text-anchor="middle">start</text>\n`;
            }
            // Draw state label
            svg += `  <text x="${pos.x}" y="${pos.y + 5}" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">${state}</text>\n`;
        }
        return svg;
    }
    generateSVGTransitions(automaton, positions, stateRadius) {
        let svg = '';
        for (const [fromState, stateTransitions] of Object.entries(automaton.transitions)) {
            const fromPos = positions.get(fromState);
            // Group transitions by target state
            const transitionGroups = new Map();
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                let targetState;
                let label;
                switch (automaton.type) {
                    case 'dfa':
                        targetState = destination;
                        label = symbol;
                        break;
                    case 'nfa':
                        for (const dest of destination) {
                            if (!transitionGroups.has(dest)) {
                                transitionGroups.set(dest, []);
                            }
                            transitionGroups.get(dest).push(symbol === 'epsilon' ? 'ε' : symbol);
                        }
                        continue;
                    case 'tm':
                        const tmDest = destination;
                        targetState = tmDest.state;
                        label = `${symbol}/${tmDest.write},${tmDest.move}`;
                        break;
                    default:
                        continue;
                }
                if (automaton.type !== 'nfa') {
                    if (!transitionGroups.has(targetState)) {
                        transitionGroups.set(targetState, []);
                    }
                    transitionGroups.get(targetState).push(label);
                }
            }
            // Draw grouped transitions
            for (const [targetState, labels] of transitionGroups) {
                const toPos = positions.get(targetState);
                const combinedLabel = labels.join(', ');
                if (fromState === targetState) {
                    // Self-loop
                    svg += this.generateSVGSelfLoop(fromPos, stateRadius, combinedLabel);
                }
                else {
                    // Regular transition
                    svg += this.generateSVGTransition(fromPos, toPos, stateRadius, combinedLabel);
                }
            }
        }
        return svg;
    }
    generateSVGTransition(fromPos, toPos, radius, label) {
        // Calculate arrow endpoints
        const dx = toPos.x - fromPos.x;
        const dy = toPos.y - fromPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const startX = fromPos.x + (dx / distance) * radius;
        const startY = fromPos.y + (dy / distance) * radius;
        const endX = toPos.x - (dx / distance) * radius;
        const endY = toPos.y - (dy / distance) * radius;
        // Calculate label position
        const labelX = (startX + endX) / 2;
        const labelY = (startY + endY) / 2 - 10;
        let svg = `  <line x1="${startX}" y1="${startY}" x2="${endX}" y2="${endY}" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>\n`;
        svg += `  <text x="${labelX}" y="${labelY}" font-family="Arial" font-size="12" text-anchor="middle" fill="red">${label}</text>\n`;
        return svg;
    }
    generateSVGSelfLoop(pos, radius, label) {
        const loopRadius = 25;
        const centerX = pos.x;
        const centerY = pos.y - radius - loopRadius;
        let svg = `  <circle cx="${centerX}" cy="${centerY}" r="${loopRadius}" fill="none" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>\n`;
        svg += `  <text x="${centerX}" y="${centerY - loopRadius - 10}" font-family="Arial" font-size="12" text-anchor="middle" fill="red">${label}</text>\n`;
        return svg;
    }
    calculateStatePositions(states, width, height, radius) {
        const positions = new Map();
        const margin = radius + 50;
        if (states.length <= 6) {
            // Arrange in a circle
            const centerX = width / 2;
            const centerY = height / 2;
            const circleRadius = Math.min(width, height) / 3;
            states.forEach((state, index) => {
                const angle = (2 * Math.PI * index) / states.length - Math.PI / 2;
                positions.set(state, {
                    x: centerX + circleRadius * Math.cos(angle),
                    y: centerY + circleRadius * Math.sin(angle)
                });
            });
        }
        else {
            // Arrange in a grid
            const cols = Math.ceil(Math.sqrt(states.length));
            const rows = Math.ceil(states.length / cols);
            const cellWidth = (width - 2 * margin) / cols;
            const cellHeight = (height - 2 * margin) / rows;
            states.forEach((state, index) => {
                const col = index % cols;
                const row = Math.floor(index / cols);
                positions.set(state, {
                    x: margin + cellWidth * (col + 0.5),
                    y: margin + cellHeight * (row + 0.5)
                });
            });
        }
        return positions;
    }
    generateDOT(automaton) {
        let dot = 'digraph automaton {\n';
        dot += '  rankdir=LR;\n';
        dot += '  size="8,5";\n';
        dot += '  node [shape = circle];\n';
        // Mark accepting states
        if (automaton.accept_states.length > 0) {
            dot += `  node [shape = doublecircle]; ${automaton.accept_states.join(' ')};\n`;
        }
        dot += '  node [shape = circle];\n';
        // Add invisible start node
        dot += '  start [shape=point];\n';
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
    generateJFLAP(automaton) {
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
        xml += '<structure>\n';
        xml += `  <type>${automaton.type === 'tm' ? 'turing' : 'fa'}</type>\n`;
        xml += '  <automaton>\n';
        // Add states with positions
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
                            const toId = stateMap.get(dest);
                            xml += `    <transition>\n`;
                            xml += `      <from>${fromId}</from>\n`;
                            xml += `      <to>${toId}</to>\n`;
                            xml += `      <read>${symbol === 'epsilon' ? '' : symbol}</read>\n`;
                            xml += `    </transition>\n`;
                        }
                        break;
                    case 'tm':
                        const tmDest = destination;
                        const toId = stateMap.get(tmDest.state);
                        xml += `    <transition>\n`;
                        xml += `      <from>${fromId}</from>\n`;
                        xml += `      <to>${toId}</to>\n`;
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
    generateLaTeX(automaton) {
        let latex = '% LaTeX TikZ representation of automaton\n';
        latex += '\\documentclass{article}\n';
        latex += '\\usepackage{tikz}\n';
        latex += '\\usetikzlibrary{automata,positioning}\n';
        latex += '\\begin{document}\n\n';
        latex += '\\begin{tikzpicture}[>=stealth,shorten >=1pt,auto,node distance=3cm]\n';
        // Define states
        const acceptingStates = new Set(automaton.accept_states);
        const states = automaton.states;
        states.forEach((state, index) => {
            const isAccepting = acceptingStates.has(state);
            const isInitial = state === automaton.start_state;
            let stateOptions = [];
            if (isInitial)
                stateOptions.push('initial');
            if (isAccepting)
                stateOptions.push('accepting');
            const options = stateOptions.length > 0 ? `[${stateOptions.join(',')}]` : '';
            // Position states in a line or grid
            const position = index === 0 ? '' : `[right of=${states[index - 1]}]`;
            latex += `  \\node[state${options}] (${state}) ${position} {$${state}$};\n`;
        });
        latex += '\n';
        // Add transitions
        latex += '  \\path[->]\n';
        for (const [fromState, stateTransitions] of Object.entries(automaton.transitions)) {
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                switch (automaton.type) {
                    case 'dfa':
                        latex += `    (${fromState}) edge node {$${symbol}$} (${destination})\n`;
                        break;
                    case 'nfa':
                        for (const dest of destination) {
                            const label = symbol === 'epsilon' ? '\\varepsilon' : symbol;
                            latex += `    (${fromState}) edge node {$${label}$} (${dest})\n`;
                        }
                        break;
                    case 'tm':
                        const tmDest = destination;
                        latex += `    (${fromState}) edge node {$${symbol}/${tmDest.write},${tmDest.move}$} (${tmDest.state})\n`;
                        break;
                }
            }
        }
        latex += '  ;\n';
        latex += '\\end{tikzpicture}\n\n';
        latex += '\\end{document}\n';
        return latex;
    }
    async exportToPNG(automaton, baseName) {
        vscode.window.showInformationMessage('PNG export requires external tools. Generating SVG instead...');
        const svgContent = this.generateSVG(automaton);
        await this.saveToFile(svgContent, `${baseName}_diagram.svg`, '.svg');
        vscode.window.showInformationMessage('To convert SVG to PNG, use online converters or tools like Inkscape.');
    }
    async exportToPDF(automaton, baseName) {
        let report = this.generatePDFReport(automaton);
        vscode.window.showInformationMessage('PDF export requires external tools. Generating LaTeX source...');
        await this.saveToFile(report, `${baseName}_report.tex`, '.tex');
        vscode.window.showInformationMessage('To generate PDF, compile the LaTeX file with pdflatex.');
    }
    generatePDFReport(automaton) {
        let latex = '\\documentclass{article}\n';
        latex += '\\usepackage{tikz}\n';
        latex += '\\usepackage{amsmath,amssymb}\n';
        latex += '\\usepackage{geometry}\n';
        latex += '\\geometry{margin=1in}\n';
        latex += '\\usetikzlibrary{automata,positioning}\n';
        latex += '\\title{Automaton Analysis Report}\n';
        latex += '\\date{\\today}\n';
        latex += '\\begin{document}\n';
        latex += '\\maketitle\n\n';
        // Automaton specification
        latex += '\\section{Automaton Specification}\n';
        latex += `\\textbf{Type:} ${automaton.type.toUpperCase()}\\\\\n`;
        latex += `\\textbf{States:} $Q = \\{${automaton.states.join(', ')}\\}$\\\\\n`;
        latex += `\\textbf{Alphabet:} $\\Sigma = \\{${automaton.alphabet.join(', ')}\\}$\\\\\n`;
        latex += `\\textbf{Start State:} $q_0 = ${automaton.start_state}$\\\\\n`;
        latex += `\\textbf{Accept States:} $F = \\{${automaton.accept_states.join(', ')}\\}$\\\\\n`;
        if (automaton.type === 'tm') {
            const tm = automaton;
            latex += `\\textbf{Tape Alphabet:} $\\Gamma = \\{${tm.tape_alphabet.join(', ')}\\}$\\\\\n`;
            latex += `\\textbf{Blank Symbol:} $\\square = ${tm.blank_symbol}$\\\\\n`;
        }
        // Transition function
        latex += '\n\\section{Transition Function}\n';
        latex += '\\begin{align}\n';
        for (const [fromState, stateTransitions] of Object.entries(automaton.transitions)) {
            for (const [symbol, destination] of Object.entries(stateTransitions)) {
                switch (automaton.type) {
                    case 'dfa':
                        latex += `\\delta(${fromState}, ${symbol}) &= ${destination}\\\\\n`;
                        break;
                    case 'nfa':
                        const destinations = destination.join(', ');
                        latex += `\\delta(${fromState}, ${symbol}) &= \\{${destinations}\\}\\\\\n`;
                        break;
                    case 'tm':
                        const tmDest = destination;
                        latex += `\\delta(${fromState}, ${symbol}) &= (${tmDest.state}, ${tmDest.write}, ${tmDest.move})\\\\\n`;
                        break;
                }
            }
        }
        latex += '\\end{align}\n\n';
        // Diagram
        latex += '\\section{State Diagram}\n';
        latex += '\\begin{center}\n';
        latex += this.generateLaTeX(automaton).split('\\begin{tikzpicture}')[1].split('\\end{tikzpicture}')[0];
        latex += '\\end{center}\n\n';
        latex += '\\end{document}\n';
        return latex;
    }
    async saveToFile(content, defaultFileName, extension) {
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(defaultFileName),
            filters: this.getFileFilters(extension)
        });
        if (uri) {
            await vscode.workspace.fs.writeFile(uri, Buffer.from(content));
            vscode.window.showInformationMessage(`Exported to ${uri.fsPath}`);
            // Ask if user wants to open the file
            const openChoice = await vscode.window.showInformationMessage('Export completed. Open the file?', 'Open', 'Open Folder');
            if (openChoice === 'Open') {
                const document = await vscode.workspace.openTextDocument(uri);
                await vscode.window.showTextDocument(document);
            }
            else if (openChoice === 'Open Folder') {
                await vscode.commands.executeCommand('revealFileInOS', uri);
            }
        }
    }
    getFileFilters(extension) {
        const filters = {
            'All files': ['*']
        };
        switch (extension) {
            case '.svg':
                filters['SVG files'] = ['svg'];
                break;
            case '.json':
                filters['JSON files'] = ['json'];
                break;
            case '.dot':
                filters['DOT files'] = ['dot'];
                break;
            case '.jff':
                filters['JFLAP files'] = ['jff'];
                filters['XML files'] = ['xml'];
                break;
            case '.tex':
                filters['LaTeX files'] = ['tex'];
                filters['Text files'] = ['txt'];
                break;
        }
        return filters;
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
exports.AutomataExporter = AutomataExporter;
//# sourceMappingURL=exporter.js.map