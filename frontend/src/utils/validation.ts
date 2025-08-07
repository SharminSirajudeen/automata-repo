import { z } from 'zod';

// Base validation schemas
export const EmailSchema = z.string()
  .email('Please enter a valid email address')
  .max(254, 'Email address is too long')
  .transform(email => email.toLowerCase().trim());

export const PasswordSchema = z.string()
  .min(8, 'Password must be at least 8 characters long')
  .max(128, 'Password is too long')
  .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 
    'Password must contain at least one lowercase letter, uppercase letter, number, and special character');

export const UsernameSchema = z.string()
  .min(3, 'Username must be at least 3 characters long')
  .max(30, 'Username cannot exceed 30 characters')
  .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, hyphens, and underscores')
  .transform(username => username.toLowerCase().trim());

// Automata validation schemas
export const StateIdSchema = z.string()
  .min(1, 'State ID is required')
  .max(50, 'State ID is too long')
  .regex(/^[a-zA-Z0-9_-]+$/, 'State ID can only contain letters, numbers, hyphens, and underscores');

export const StateSchema = z.object({
  id: StateIdSchema,
  name: z.string().max(100, 'State name is too long').optional(),
  x: z.number().min(0).max(10000, 'X coordinate is out of bounds'),
  y: z.number().min(0).max(10000, 'Y coordinate is out of bounds'),
  isStart: z.boolean(),
  isAccept: z.boolean(),
  isReject: z.boolean().optional(),
  metadata: z.record(z.any()).optional()
});

export const TransitionSymbolSchema = z.string()
  .max(10, 'Transition symbol is too long')
  .refine(symbol => symbol === 'ε' || symbol === 'epsilon' || /^[a-zA-Z0-9_#$@!%*?&+-]$/.test(symbol), 
    'Invalid transition symbol');

export const TransitionSchema = z.object({
  id: z.string().min(1, 'Transition ID is required').max(100),
  from: StateIdSchema,
  to: StateIdSchema,
  symbol: TransitionSymbolSchema,
  // PDA-specific fields
  popSymbol: z.string().max(10).optional(),
  pushSymbol: z.string().max(10).optional(),
  // TM-specific fields
  writeSymbol: z.string().max(1).optional(),
  direction: z.enum(['L', 'R', 'S']).optional(),
  metadata: z.record(z.any()).optional()
});

export const AutomataTypeSchema = z.enum(['dfa', 'nfa', 'pda', 'tm', 'cfg']);

export const AutomatonSchema = z.object({
  id: z.string().min(1, 'Automaton ID is required').max(100),
  name: z.string().min(1, 'Automaton name is required').max(200),
  type: AutomataTypeSchema,
  description: z.string().max(1000, 'Description is too long').optional(),
  alphabet: z.array(z.string().max(10)).max(50, 'Alphabet is too large'),
  states: z.array(StateSchema).min(1, 'At least one state is required').max(100, 'Too many states'),
  transitions: z.array(TransitionSchema).max(500, 'Too many transitions'),
  startState: StateIdSchema,
  acceptStates: z.array(StateIdSchema).max(50, 'Too many accept states'),
  stackAlphabet: z.array(z.string().max(10)).max(50, 'Stack alphabet is too large').optional(),
  tapeAlphabet: z.array(z.string().max(10)).max(50, 'Tape alphabet is too large').optional(),
  metadata: z.record(z.any()).optional()
});

// Test string validation
export const TestStringSchema = z.string()
  .max(1000, 'Test string is too long')
  .refine(str => /^[a-zA-Z0-9ε]*$/.test(str), 'Test string contains invalid characters');

export const TestCaseSchema = z.object({
  input: TestStringSchema,
  expectedResult: z.enum(['accept', 'reject']),
  description: z.string().max(500).optional()
});

// User input validation
export const UserInputSchema = z.object({
  email: EmailSchema.optional(),
  username: UsernameSchema.optional(),
  password: PasswordSchema.optional(),
  firstName: z.string().max(50, 'First name is too long').optional(),
  lastName: z.string().max(50, 'Last name is too long').optional()
});

// Settings validation
export const AnimationSettingsSchema = z.object({
  duration: z.number().min(100).max(10000, 'Animation duration must be between 100ms and 10s'),
  easing: z.enum(['linear', 'easeIn', 'easeOut', 'easeInOut', 'wobbly']),
  autoPlay: z.boolean(),
  loop: z.boolean(),
  showDetails: z.boolean(),
  highlightPath: z.boolean(),
  animationSpeed: z.number().min(0.1).max(5, 'Animation speed must be between 0.1x and 5x')
});

export const ThemeSettingsSchema = z.object({
  mode: z.enum(['light', 'dark', 'system']),
  primaryColor: z.string().regex(/^#[0-9A-F]{6}$/i, 'Invalid color format'),
  fontSize: z.enum(['small', 'medium', 'large']),
  compactMode: z.boolean(),
  highContrast: z.boolean()
});

// API validation schemas
export const CreateAutomatonRequestSchema = z.object({
  automaton: AutomatonSchema,
  isPublic: z.boolean().default(false),
  tags: z.array(z.string().max(30)).max(10, 'Too many tags').optional()
});

export const UpdateAutomatonRequestSchema = CreateAutomatonRequestSchema.partial();

export const SimulateRequestSchema = z.object({
  automatonId: z.string().min(1, 'Automaton ID is required'),
  input: TestStringSchema,
  stepByStep: z.boolean().default(false),
  maxSteps: z.number().min(1).max(10000).default(1000)
});

export const SearchRequestSchema = z.object({
  query: z.string().max(200, 'Search query is too long'),
  type: AutomataTypeSchema.optional(),
  limit: z.number().min(1).max(100).default(20),
  offset: z.number().min(0).default(0),
  sortBy: z.enum(['name', 'created', 'modified', 'popularity']).default('name'),
  sortOrder: z.enum(['asc', 'desc']).default('asc')
});

// File upload validation
export const FileUploadSchema = z.object({
  file: z.instanceof(File)
    .refine(file => file.size <= 10 * 1024 * 1024, 'File size must be less than 10MB')
    .refine(file => ['application/json', 'text/xml', 'text/plain'].includes(file.type), 
      'Only JSON, XML, and text files are allowed'),
  automatonType: AutomataTypeSchema
});

// Bulk operations validation
export const BulkDeleteSchema = z.object({
  ids: z.array(z.string().min(1)).min(1, 'At least one ID is required').max(100, 'Too many IDs')
});

export const BulkUpdateSchema = z.object({
  ids: z.array(z.string().min(1)).min(1, 'At least one ID is required').max(100, 'Too many IDs'),
  updates: z.object({
    isPublic: z.boolean().optional(),
    tags: z.array(z.string().max(30)).max(10).optional()
  })
});

// Validation utility functions
export const validateEmail = (email: string) => {
  try {
    return { success: true as const, data: EmailSchema.parse(email), error: null };
  } catch (error) {
    return { success: false as const, data: null, error: error as z.ZodError };
  }
};

export const validatePassword = (password: string) => {
  try {
    return { success: true as const, data: PasswordSchema.parse(password), error: null };
  } catch (error) {
    return { success: false as const, data: null, error: error as z.ZodError };
  }
};

export const validateAutomaton = (automaton: unknown) => {
  try {
    return { success: true as const, data: AutomatonSchema.parse(automaton), error: null };
  } catch (error) {
    return { success: false as const, data: null, error: error as z.ZodError };
  }
};

export const validateTestString = (input: string) => {
  try {
    return { success: true as const, data: TestStringSchema.parse(input), error: null };
  } catch (error) {
    return { success: false as const, data: null, error: error as z.ZodError };
  }
};

// Advanced validation helpers
export const createCustomValidator = <T>(schema: z.ZodSchema<T>) => {
  return (data: unknown) => {
    try {
      return { success: true as const, data: schema.parse(data), error: null };
    } catch (error) {
      return { success: false as const, data: null, error: error as z.ZodError };
    }
  };
};

export const sanitizeInput = (input: string, maxLength: number = 1000): string => {
  return input
    .trim()
    .slice(0, maxLength)
    .replace(/[<>\"'&]/g, (match) => {
      const entities: { [key: string]: string } = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '&': '&amp;'
      };
      return entities[match];
    });
};

export const validateAndSanitizeInput = (input: string, schema: z.ZodSchema<string>) => {
  const sanitized = sanitizeInput(input);
  return createCustomValidator(schema)(sanitized);
};

// Rate limiting validation
export const RateLimitSchema = z.object({
  endpoint: z.string(),
  limit: z.number().min(1).max(10000),
  windowMs: z.number().min(1000).max(3600000), // 1 second to 1 hour
  skipSuccessfulRequests: z.boolean().default(false)
});

// Environment configuration validation
export const EnvironmentConfigSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']),
  API_URL: z.string().url('Invalid API URL'),
  WEBSOCKET_URL: z.string().url('Invalid WebSocket URL'),
  SENTRY_DSN: z.string().url('Invalid Sentry DSN').optional(),
  ENABLE_ANALYTICS: z.boolean().default(false)
});

export default {
  EmailSchema,
  PasswordSchema,
  UsernameSchema,
  StateSchema,
  TransitionSchema,
  AutomatonSchema,
  TestStringSchema,
  validateEmail,
  validatePassword,
  validateAutomaton,
  validateTestString,
  sanitizeInput,
  validateAndSanitizeInput
};