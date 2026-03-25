// src/lib/writing/types.ts

import type { BlockInstance } from "./blocks";

export type Genre = "essay" | "article" | "academic" | "creative" | "business";
export type Trait = "ideas" | "organization" | "voice" | "wordChoice" | "fluency" | "conventions" | "presentation";
export type Severity = "good" | "question" | "suggestion" | "issue";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface StepCard {
  id: string;
  stepIndex: number;
  totalSteps: number;
  title: string;
  mnemonic?: string;
  instructions: string;
  checklist?: string[];
  example?: string;
  completed: boolean;
}

export interface Annotation {
  id: string;
  paragraph: number;
  startOffset: number;
  endOffset: number;
  trait: Trait;
  severity: Severity;
  message: string;
  rewrite?: string;
}

export interface IdeaCheckpointOutput {
  userInput: string;
}

export interface DraftOutput {
  document: string; // Plain text with \n\n paragraph breaks
}

export interface GrammarOutput {
  document: string; // Plain text (corrected)
  corrections: Array<{
    original: string;
    corrected: string;
    reason: string;
  }>;
}

export type PipelineOutputs = {
  thesis?: IdeaCheckpointOutput;
  outline?: IdeaCheckpointOutput;
  hook?: IdeaCheckpointOutput;
  draft?: DraftOutput;
  analyze?: AnalyzeResponse;
  grammar?: GrammarOutput;
};

export type PipelineStatus = 'idle' | 'executing' | 'checkpoint' | 'done' | 'cancelled' | 'error';

export interface PipelineState {
  blocks: BlockInstance[];
  currentIndex: number;
  outputs: PipelineOutputs;
  status: PipelineStatus;
  document: string;
  language: 'en' | 'zh';
  executionLog: Array<{
    blockType: string;
    dialogue: string;
    timestamp: number;
  }>;
}

export interface DailyTip {
  id: string;
  trait: Trait;
  tip: string;
  example?: { before: string; after: string };
  exercisePrompt?: string;
}

export interface LabExample {
  id: string;
  topic: string;
  coldText: string;
  humanWarmText: string;
  humanExplanation: string;
  teachingPoint: string;
  focusTrait: Trait;
}

export interface AnalysisSnapshot {
  date: string;
  genre: Genre;
  wordCount: number;
  traitScores: Record<Trait, number>;
  annotationCounts: Record<Severity, number>;
}

export interface WriterProfile {
  userId: string;
  genreExperience: Record<Genre, number>;
  analysisHistory: AnalysisSnapshot[];
  traitScores: Record<Trait, { date: string; score: number }[]>;
  streak: { current: number; longest: number; lastActiveDate: string };
  completedExercises: string[];
  stats: { totalWords: number; totalSessions: number; totalAnalyses: number };
  preferences: { showDailyTips: boolean };
}

export interface WritingCenterState {
  draft: {
    genre: Genre;
    topic: string;
    document: string;
    messages: ChatMessage[];
    annotations: Annotation[];
    lastSaved: number;
  };
  profile: WriterProfile;
}

// -- API Request/Response --

export interface WritingAssistRequest {
  action: "guide" | "analyze" | "expand" | "daily-tip" | "lab-rewrite" | "report" | "auto-execute";
  mode?: "step" | "dialogue";
  genre?: Genre;
  topic?: string;
  document?: string;
  messages?: ChatMessage[];
  annotationId?: string;
  annotationContext?: Annotation;
  experienceLevel?: number;
  traitScores?: Record<Trait, number>;
  analysisHistory?: AnalysisSnapshot[];
  text?: string;
  temperatures?: number[];
  blockSystemPrompt?: string;
  profile?: ReportProfileData;
  blockType?: "draft" | "analyze" | "grammar";
  language?: 'en' | 'zh';
  previousOutputs?: PipelineOutputs;
}

export interface AutoExecuteRequest {
  action: "auto-execute";
  blockType: "draft" | "analyze" | "grammar";
  genre: string;
  topic: string;
  language: 'en' | 'zh';
  document?: string;
  previousOutputs?: PipelineOutputs;
}

export interface AutoExecuteResponse {
  dialogue: string;
  documentUpdate?: string;
  analysisResult?: AnalyzeResponse;
  grammarResult?: GrammarOutput;
}

export interface GuideStepResponse {
  type: "step";
  cards: StepCard[];
}

export interface GuideDialogueResponse {
  type: "dialogue";
  message: string;
}

export interface AnalyzeResponse {
  annotations: Annotation[];
  traitScores: Record<Trait, number>;
  summary: string;
  conventionsSuppressed: boolean;
}

export interface ExpandResponse {
  detail: string;
  suggestion?: string;
  question: string;
}

export interface DailyTipResponse {
  tip: DailyTip;
}

export interface LabRewriteResponse {
  rewrites: { temperature: number; text: string; explanation: string }[];
}

export interface ReportProfileData {
  recentAnalyses: AnalysisSnapshot[];
  traitTrends: Record<Trait, number[]>;
  streak: number;
  totalWordsThisWeek: number;
  genresThisWeek: string[];
}

export interface ReportResponse {
  summary: string;
  improvements: string;
  weakPoints: string;
  nextWeekFocus: string;
  encouragement: string;
}
