// Automata Visualization Script

let currentAutomaton = null;
let simulationState = {
    isRunning: false,
    currentStates: new Set(),
    inputString: '',
    position: 0,
    tape: [],
    tapePosition: 0,
    history: []
};

function initializeVisualization(automaton) {
    currentAutomaton = automaton;
    drawAutomaton(automaton);
    setupEventHandlers();
}

function drawAutomaton(automaton) {
    const container = d3.select('#automata-diagram');
    container.selectAll('*').remove();
    
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    
    const svg = container
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    const g = svg.append('g');
    
    // Create force simulation
    const simulation = d3.forceSimulation()
        .force('link', d3.forceLink().id(d => d.id).distance(150))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(50));
    
    // Prepare data
    const nodes = automaton.states.map(state => ({
        id: state,
        isStart: state === automaton.start_state,
        isAccepting: automaton.accept_states.includes(state),
        isRejecting: automaton.reject_states && automaton.reject_states.includes(state)
    }));
    
    const links = [];
    for (const [fromState, transitions] of Object.entries(automaton.transitions)) {
        for (const [symbol, destination] of Object.entries(transitions)) {
            switch (automaton.type) {
                case 'dfa':
                    links.push({
                        source: fromState,
                        target: destination,
                        symbol: symbol
                    });
                    break;
                case 'nfa':
                    for (const dest of destination) {
                        links.push({
                            source: fromState,
                            target: dest,
                            symbol: symbol
                        });
                    }
                    break;
                case 'tm':
                    links.push({
                        source: fromState,
                        target: destination.state,
                        symbol: symbol,
                        write: destination.write,
                        move: destination.move
                    });
                    break;
            }
        }
    }
    
    // Group links by source-target pairs
    const linkGroups = {};
    links.forEach(link => {
        const key = `${link.source}-${link.target}`;
        if (!linkGroups[key]) {
            linkGroups[key] = [];
        }
        linkGroups[key].push(link);
    });
    
    const processedLinks = [];
    for (const [key, group] of Object.entries(linkGroups)) {
        if (group.length === 1) {
            processedLinks.push({
                ...group[0],
                labels: [formatTransitionLabel(group[0], automaton.type)]
            });
        } else {
            processedLinks.push({
                source: group[0].source,
                target: group[0].target,
                labels: group.map(link => formatTransitionLabel(link, automaton.type))
            });
        }
    });
    
    // Create arrow markers
    svg.defs = svg.append('defs');
    
    svg.defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#666');
    
    // Draw links
    const link = g.append('g')
        .selectAll('path')
        .data(processedLinks)
        .enter().append('path')
        .attr('class', 'link')
        .attr('fill', 'none')
        .attr('stroke', '#666')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)');
    
    // Draw link labels
    const linkLabels = g.append('g')
        .selectAll('text')
        .data(processedLinks)
        .enter().append('text')
        .attr('class', 'link-label')
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#333')
        .attr('background', 'white')
        .text(d => d.labels.join(', '));
    
    // Draw nodes
    const node = g.append('g')
        .selectAll('g')
        .data(nodes)
        .enter().append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // Add circles for states
    node.append('circle')
        .attr('r', 25)
        .attr('fill', d => {
            if (d.isStart && d.isAccepting) return '#4CAF50';
            if (d.isAccepting) return '#2196F3';
            if (d.isRejecting) return '#F44336';
            return '#9E9E9E';
        })
        .attr('stroke', d => d.isAccepting ? '#1976D2' : '#666')
        .attr('stroke-width', d => d.isAccepting ? 3 : 2);
    
    // Add double circle for accepting states
    node.filter(d => d.isAccepting)
        .append('circle')
        .attr('r', 20)
        .attr('fill', 'none')
        .attr('stroke', '#1976D2')
        .attr('stroke-width', 2);
    
    // Add start arrow
    node.filter(d => d.isStart)
        .append('path')
        .attr('d', 'M -45 -5 L -30 0 L -45 5 Z')
        .attr('fill', '#333')
        .attr('stroke', '#333');
    
    // Add state labels
    node.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .attr('fill', 'white')
        .text(d => d.id);
    
    simulation
        .nodes(nodes)
        .on('tick', ticked);
    
    simulation.force('link')
        .links(processedLinks);
    
    function ticked() {
        link.attr('d', d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dr = Math.sqrt(dx * dx + dy * dy);
            
            // Calculate curve for self-loops
            if (d.source === d.target) {
                return `M ${d.source.x} ${d.source.y - 25} A 25 25 0 1 1 ${d.source.x + 1} ${d.source.y - 25}`;
            }
            
            // Calculate curve for regular links
            const curve = dr * 0.3;
            return `M ${d.source.x} ${d.source.y} A ${curve} ${curve} 0 0 1 ${d.target.x} ${d.target.y}`;
        });
        
        linkLabels
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2 - 5);
        
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    }
    
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function formatTransitionLabel(link, type) {
    switch (type) {
        case 'dfa':
        case 'nfa':
            return link.symbol === 'epsilon' || link.symbol === 'ε' || link.symbol === 'λ' 
                ? 'ε' : link.symbol;
        case 'tm':
            return `${link.symbol} → ${link.write}, ${link.move}`;
        default:
            return link.symbol;
    }
}

function setupEventHandlers() {
    // Handle window resize
    window.addEventListener('resize', () => {
        if (currentAutomaton) {
            drawAutomaton(currentAutomaton);
        }
    });
}

function simulateStep(automaton, input, currentStates, position) {
    const nextStates = new Set();
    
    if (position >= input.length) {
        return { states: currentStates, position, completed: true };
    }
    
    const symbol = input[position];
    
    for (const state of currentStates) {
        const transitions = automaton.transitions[state];
        if (transitions && transitions[symbol]) {
            switch (automaton.type) {
                case 'dfa':
                    nextStates.add(transitions[symbol]);
                    break;
                case 'nfa':
                    for (const nextState of transitions[symbol]) {
                        nextStates.add(nextState);
                    }
                    break;
                case 'tm':
                    // Turing machine simulation is more complex
                    const transition = transitions[symbol];
                    return simulateTuringMachine(automaton, input, state, position, transition);
            }
        }
    }
    
    // Handle epsilon transitions for NFA
    if (automaton.type === 'nfa') {
        const epsilonClosure = new Set(nextStates);
        const queue = [...nextStates];
        
        while (queue.length > 0) {
            const currentState = queue.shift();
            const transitions = automaton.transitions[currentState];
            
            if (transitions) {
                for (const epsilonSymbol of ['epsilon', 'ε', 'λ']) {
                    if (transitions[epsilonSymbol]) {
                        for (const nextState of transitions[epsilonSymbol]) {
                            if (!epsilonClosure.has(nextState)) {
                                epsilonClosure.add(nextState);
                                queue.push(nextState);
                            }
                        }
                    }
                }
            }
        }
        
        return { states: epsilonClosure, position: position + 1, completed: false };
    }
    
    return { states: nextStates, position: position + 1, completed: false };
}

function simulateTuringMachine(tm, input, currentState, tapePosition, transition) {
    if (!transition) {
        return { state: currentState, position: tapePosition, tape: simulationState.tape, halted: true };
    }
    
    // Update tape
    const newTape = [...simulationState.tape];
    newTape[tapePosition] = transition.write;
    
    // Move tape head
    let newPosition = tapePosition;
    if (transition.move === 'L') {
        newPosition = Math.max(0, tapePosition - 1);
    } else if (transition.move === 'R') {
        newPosition = tapePosition + 1;
        // Extend tape if necessary
        while (newTape.length <= newPosition) {
            newTape.push(tm.blank_symbol);
        }
    }
    
    return {
        state: transition.state,
        position: newPosition,
        tape: newTape,
        halted: tm.accept_states.includes(transition.state) || 
                (tm.reject_states && tm.reject_states.includes(transition.state))
    };
}

function updateVisualization() {
    // Highlight current states
    const nodes = d3.selectAll('.node circle');
    nodes.attr('stroke-width', 2);
    
    if (simulationState.currentStates.size > 0) {
        nodes.filter(d => simulationState.currentStates.has(d.id))
            .attr('stroke-width', 4)
            .attr('stroke', '#FF5722');
    }
    
    // Update status panel
    const statusPanel = document.getElementById('simulation-status');
    statusPanel.classList.remove('hidden');
    
    document.getElementById('current-state').textContent = 
        `Current: {${Array.from(simulationState.currentStates).join(', ')}}`;
    
    document.getElementById('input-position').textContent = 
        `Position: ${simulationState.position}/${simulationState.inputString.length}`;
    
    // Update result
    const resultElement = document.getElementById('simulation-result');
    if (simulationState.position >= simulationState.inputString.length) {
        const hasAcceptingState = Array.from(simulationState.currentStates)
            .some(state => currentAutomaton.accept_states.includes(state));
        
        resultElement.textContent = hasAcceptingState ? 'ACCEPTED' : 'REJECTED';
        resultElement.className = hasAcceptingState ? 'result accepted' : 'result rejected';
    } else {
        resultElement.textContent = 'RUNNING';
        resultElement.className = 'result';
    }
    
    // Update tape for Turing machines
    if (currentAutomaton.type === 'tm') {
        updateTapeView();
    }
}

function updateTapeView() {
    const tapeView = document.getElementById('tape-view');
    if (!tapeView) return;
    
    tapeView.innerHTML = '<h4>Tape</h4>';
    
    const cellsContainer = document.createElement('div');
    cellsContainer.className = 'tape-cells';
    
    const startPos = Math.max(0, simulationState.tapePosition - 5);
    const endPos = Math.min(simulationState.tape.length, simulationState.tapePosition + 6);
    
    for (let i = startPos; i < endPos; i++) {
        const cell = document.createElement('div');
        cell.className = 'tape-cell';
        
        if (i === simulationState.tapePosition) {
            cell.classList.add('current');
        }
        
        const symbol = simulationState.tape[i] || currentAutomaton.blank_symbol;
        cell.textContent = symbol;
        
        if (symbol === currentAutomaton.blank_symbol) {
            cell.classList.add('blank');
        }
        
        cellsContainer.appendChild(cell);
    }
    
    tapeView.appendChild(cellsContainer);
}

// Export functions for external use
window.automataPreview = {
    initializeVisualization,
    drawAutomaton,
    simulateStep,
    updateVisualization
};