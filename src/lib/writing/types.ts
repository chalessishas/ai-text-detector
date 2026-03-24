// src/lib/writing/types.ts

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
  action: "guide" | "analyze" | "expand" | "daily-tip" | "lab-rewrite" | "report";
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
