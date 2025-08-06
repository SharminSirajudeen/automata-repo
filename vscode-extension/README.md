# Automata Development Extension

A comprehensive VS Code extension for developing, visualizing, and simulating finite automata, pushdown automata, and Turing machines.

## Features

### 🎯 Core Functionality
- **Syntax Highlighting**: Rich syntax highlighting for `.dfa`, `.nfa`, and `.tm` files
- **IntelliSense**: Smart completion for automata definitions with context-aware suggestions
- **Real-time Validation**: Instant error checking and semantic validation
- **Live Preview**: Interactive visual representation of automata with D3.js-powered diagrams

### 🔧 Automata Operations
- **Simulation**: Step-by-step execution with input string testing
- **DFA Minimization**: Automatic state minimization with detailed analysis
- **Format Conversion**: Convert between DFA, NFA, and other formats
- **Export Options**: Export to SVG, PNG, DOT, JFLAP XML, LaTeX, and PDF

### 📊 Visualization & Analysis
- **Interactive Diagrams**: Drag-and-drop state positioning with zoom/pan support
- **Simulation Controls**: Play, step, reset controls for tracing execution
- **Tape Visualization**: Special tape view for Turing machine simulations
- **State Analysis**: Detailed information about reachable, dead, and equivalent states

### 🚀 Advanced Features
- **Language Server**: Full language server protocol implementation
- **Error Diagnostics**: Comprehensive error reporting with quick fixes
- **Hover Information**: Contextual help and documentation
- **Go to Definition**: Navigate between state definitions
- **Code Templates**: Quick scaffolding for common automata patterns

## File Types Supported

### DFA Files (`.dfa`)
```json
{
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start_state": "q0",
  "accept_states": ["q2"],
  "transitions": {
    "q0": {
      "0": "q1",
      "1": "q0"
    },
    "q1": {
      "0": "q2",
      "1": "q0"
    },
    "q2": {
      "0": "q2",
      "1": "q2"
    }
  }
}
```

### NFA Files (`.nfa`)
```json
{
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start_state": "q0",
  "accept_states": ["q2"],
  "transitions": {
    "q0": {
      "0": ["q0", "q1"],
      "1": ["q0"],
      "epsilon": ["q1"]
    },
    "q1": {
      "1": ["q2"]
    },
    "q2": {}
  }
}
```

### Turing Machine Files (`.tm`)
```json
{
  "states": ["q0", "q1", "qaccept", "qreject"],
  "alphabet": ["0", "1"],
  "tape_alphabet": ["0", "1", "_"],
  "blank_symbol": "_",
  "start_state": "q0",
  "accept_states": ["qaccept"],
  "reject_states": ["qreject"],
  "transitions": {
    "q0": {
      "0": {
        "state": "q1",
        "write": "X",
        "move": "R"
      }
    }
  }
}
```

## Commands

| Command | Description | Keybinding |
|---------|-------------|------------|
| `Show Automata Preview` | Open live preview panel | `Ctrl+K V` |
| `Simulate Automata` | Run simulation with input | `Ctrl+Shift+R` |
| `Test Input String` | Test specific input string | - |
| `Minimize DFA` | Minimize DFA states | - |
| `Convert Automata Format` | Convert between formats | - |
| `Validate Automata` | Run validation checks | - |
| `Export Automata` | Export to various formats | - |

## Installation

### From VSIX Package
1. Download the `.vsix` file
2. Open VS Code
3. Run `Extensions: Install from VSIX...` from the Command Palette
4. Select the downloaded file

### From Source
1. Clone this repository
2. Run `npm install` in the extension directory
3. Run `npm run compile` to build
4. Press `F5` to open Extension Development Host

## Configuration

Configure the extension through VS Code settings:

```json
{
  "automata.preview.autoRefresh": true,
  "automata.simulation.speed": 1000,
  "automata.validation.enableRealtime": true,
  "automata.export.format": "svg"
}
```

## Usage Examples

### Creating a New Automaton
1. Use Command Palette: `Automata: New DFA/NFA/TM`
2. Choose template and filename
3. Edit the generated JSON structure

### Simulating an Automaton
1. Open any `.dfa`, `.nfa`, or `.tm` file
2. Click the preview button or press `Ctrl+K V`
3. Enter input string and click "Simulate"
4. Use step controls to trace execution

### Converting Formats
1. Right-click in automaton file
2. Select "Convert Automata Format"
3. Choose target format
4. New file will be created with converted format

### Exporting Diagrams
1. Open preview panel
2. Choose export format from dropdown
3. Click "Export" button
4. Save to desired location

## Validation Features

The extension provides comprehensive validation:

- **Structural Validation**: Required fields, proper JSON format
- **Semantic Validation**: State references, alphabet consistency
- **Reachability Analysis**: Unreachable and dead states
- **Completeness Checks**: Missing transitions, trap states
- **Type-Specific Validation**: DFA determinism, TM tape alphabet

## Advanced Features

### Language Server Integration
- Full LSP implementation with hover, completion, and diagnostics
- Real-time error checking as you type
- Context-sensitive help and documentation

### Simulation Engine
- Deterministic and non-deterministic execution
- Epsilon closure computation for NFAs
- Turing machine tape management
- Comprehensive execution traces

### Minimization Algorithm
- Table-filling algorithm for DFA minimization
- Unreachable and dead state removal
- Equivalent state identification
- Detailed minimization analysis

## Export Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| SVG | `.svg` | Scalable vector graphics |
| PNG | `.png` | Raster image (via conversion) |
| DOT | `.dot` | GraphViz format |
| JFLAP | `.jff` | JFLAP XML format |
| LaTeX | `.tex` | TikZ diagram code |
| JSON | `.json` | Raw automaton data |
| PDF | `.pdf` | Comprehensive report |

## Development

### Building from Source
```bash
git clone <repository>
cd vscode-extension
npm install
npm run compile
```

### Running Tests
```bash
npm run test
```

### Packaging
```bash
npm run package
```

## Contributing

Contributions are welcome! Please see the contributing guidelines in the main repository.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Changelog

### Version 1.0.0
- Initial release
- Complete automata support (DFA, NFA, TM)
- Live preview and simulation
- Export functionality
- Language server integration
- Comprehensive validation

## Support

For issues, questions, or feature requests, please visit the GitHub repository issues page.

---

**Enjoy developing automata with this powerful VS Code extension!** 🎉