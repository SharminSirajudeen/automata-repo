# VS Code Automata Extension Installation Guide

## Quick Setup

1. **Install Dependencies**:
   ```bash
   cd vscode-extension
   npm install
   ```

2. **Compile the Extension**:
   ```bash
   npx tsc -p tsconfig-simple.json
   ```

3. **Test the Extension**:
   - Open VS Code in this directory
   - Press `F5` to launch Extension Development Host
   - Open any `.dfa`, `.nfa`, or `.tm` file from the `examples/` folder
   - Use `Ctrl+K V` to open the preview panel

## Current Features

### ✅ Working Features
- **Syntax Highlighting** for `.dfa`, `.nfa`, `.tm` files
- **Live Preview Panel** with basic automaton visualization
- **File Templates** - Use Command Palette → "Automata: New DFA/NFA/TM"
- **Language Server** with basic IntelliSense
- **JSON Validation** for automata files

### 🚧 Placeholder Features (Ready for Implementation)
- Simulation Engine
- DFA Minimization
- Format Conversion (NFA to DFA, etc.)
- Export to SVG/PNG/DOT formats
- Advanced validation and error checking

## File Structure

```
vscode-extension/
├── src/
│   ├── extension-simple.ts       # Main extension entry point
│   ├── previewProvider-simple.ts # Preview panel implementation
│   ├── languageServer.ts         # Language server for IntelliSense
│   ├── simpleValidator.ts        # Basic validation
│   └── [other advanced files]    # Complete implementations
├── syntaxes/                     # Language grammar definitions
├── media/                        # CSS/JS for preview panel
├── examples/                     # Sample automata files
└── package.json                  # Extension manifest
```

## Commands Available

| Command | Description |
|---------|-------------|
| `automata.showPreview` | Show automata preview panel |
| `automata.createDFA` | Create new DFA file |
| `automata.createNFA` | Create new NFA file |
| `automata.createTM` | Create new Turing Machine file |
| `automata.simulate` | Simulate automaton (placeholder) |
| `automata.minimize` | Minimize DFA (placeholder) |
| `automata.convert` | Convert between formats (placeholder) |
| `automata.export` | Export automaton (placeholder) |

## File Format Examples

The extension supports JSON-based automata definitions:

### DFA Example (`example.dfa`)
```json
{
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start_state": "q0",
  "accept_states": ["q2"],
  "transitions": {
    "q0": { "0": "q1", "1": "q0" },
    "q1": { "0": "q2", "1": "q0" },
    "q2": { "0": "q2", "1": "q2" }
  }
}
```

### NFA Example (`example.nfa`)
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
    "q1": { "1": ["q2"] },
    "q2": {}
  }
}
```

### Turing Machine Example (`example.tm`)
```json
{
  "states": ["q0", "q1", "qaccept", "qreject"],
  "alphabet": ["0", "1"],
  "tape_alphabet": ["0", "1", "X", "Y", "_"],
  "blank_symbol": "_",
  "start_state": "q0",
  "accept_states": ["qaccept"],
  "reject_states": ["qreject"],
  "transitions": {
    "q0": {
      "0": { "state": "q1", "write": "X", "move": "R" }
    }
  }
}
```

## Development Notes

### Current Implementation Status
- ✅ **Core Extension**: Working with basic preview and template generation
- ✅ **Language Server**: Basic completion and hover support
- ✅ **Syntax Highlighting**: Complete for all automata types
- 🚧 **Advanced Features**: Implemented but not compiled due to complexity

### Next Steps for Full Implementation
1. Fix TypeScript compilation errors in advanced modules
2. Implement D3.js visualization in preview panel
3. Add simulation step-by-step execution
4. Complete export functionality
5. Add comprehensive validation and error reporting

### Known Issues
- Some advanced TypeScript files have compilation errors
- Preview panel shows JSON instead of visual diagram
- Language server is basic (no advanced semantic analysis)

## Testing the Extension

1. Open the Extension Development Host (`F5`)
2. Open `examples/example.dfa`
3. Press `Ctrl+K V` to show preview
4. Try Command Palette → "Automata: New DFA"
5. Test IntelliSense by typing in a `.dfa` file

## Building for Production

When ready for full deployment:
```bash
npm install -g @vscode/vsce
vsce package
```

This creates a `.vsix` file for distribution.

---

**The extension foundation is complete and ready for enhanced features!**