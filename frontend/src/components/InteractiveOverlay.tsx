import React from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { State, Transition } from '../types/automata';

interface InteractiveOverlayProps {
  states: State[];
  transitions: Transition[];
  canvasWidth: number;
  canvasHeight: number;
  stateRadius: number;
  stepExplanations?: { [key: string]: string };
  onStateHover?: (stateId: string) => void;
  onTransitionHover?: (transitionIndex: number) => void;
}

export const InteractiveOverlay: React.FC<InteractiveOverlayProps> = ({
  states,
  transitions,
  canvasWidth,
  canvasHeight,
  stateRadius,
  stepExplanations = {},
  onStateHover,
  onTransitionHover,
}) => {
  return (
    <TooltipProvider>
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{ width: canvasWidth, height: canvasHeight }}
      >
        {states.map((state) => (
          <Tooltip key={state.id}>
            <TooltipTrigger asChild>
              <div
                className="absolute pointer-events-auto cursor-pointer rounded-full"
                style={{
                  left: state.x - stateRadius,
                  top: state.y - stateRadius,
                  width: stateRadius * 2,
                  height: stateRadius * 2,
                }}
                onMouseEnter={() => onStateHover?.(state.id)}
              />
            </TooltipTrigger>
            <TooltipContent>
              <div className="max-w-xs">
                <p className="font-semibold">State {state.id}</p>
                <p className="text-sm">
                  {state.is_start && "Start state - "}
                  {state.is_accept && "Accept state - "}
                  {stepExplanations[state.id] || "Click for more details"}
                </p>
              </div>
            </TooltipContent>
          </Tooltip>
        ))}
        
        {transitions.map((transition, index) => {
          const fromState = states.find(s => s.id === transition.from_state);
          const toState = states.find(s => s.id === transition.to_state);
          if (!fromState || !toState) return null;
          
          const midX = (fromState.x + toState.x) / 2;
          const midY = (fromState.y + toState.y) / 2;
          
          return (
            <Tooltip key={index}>
              <TooltipTrigger asChild>
                <div
                  className="absolute pointer-events-auto cursor-pointer"
                  style={{
                    left: midX - 15,
                    top: midY - 15,
                    width: 30,
                    height: 30,
                  }}
                  onMouseEnter={() => onTransitionHover?.(index)}
                />
              </TooltipTrigger>
              <TooltipContent>
                <div className="max-w-xs">
                  <p className="font-semibold">Transition: {transition.symbol}</p>
                  <p className="text-sm">From {transition.from_state} to {transition.to_state}</p>
                  <p className="text-sm">{stepExplanations[`transition_${index}`] || "Part of the automaton logic"}</p>
                </div>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </TooltipProvider>
  );
};
