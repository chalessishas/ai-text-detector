import type { WritingCenterState, WriterProfile, Genre, AnalysisSnapshot, Trait } from './types';

const STORAGE_KEY = 'writing-center';

const TRAITS: Trait[] = ['ideas', 'organization', 'voice', 'wordChoice', 'fluency', 'conventions', 'presentation'];

export function createDefaultProfile(): WriterProfile {
  return {
    userId: '',
    genreExperience: { essay: 0, article: 0, academic: 0, creative: 0, business: 0 },
    analysisHistory: [],
    traitScores: {
      ideas: [], organization: [], voice: [], wordChoice: [],
      fluency: [], conventions: [], presentation: [],
    },
    streak: { current: 0, longest: 0, lastActiveDate: '' },
    completedExercises: [],
    stats: { totalWords: 0, totalSessions: 0, totalAnalyses: 0 },
    preferences: { showDailyTips: true },
  };
}

export function createDefaultState(): WritingCenterState {
  return {
    draft: {
      genre: 'essay',
      topic: '',
      document: '',
      messages: [],
      annotations: [],
      lastSaved: 0,
    },
    profile: createDefaultProfile(),
  };
}

export function loadState(): WritingCenterState {
  if (typeof window === 'undefined') return createDefaultState();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return createDefaultState();
    return JSON.parse(raw) as WritingCenterState;
  } catch {
    return createDefaultState();
  }
}

export function saveState(state: WritingCenterState): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // quota exceeded — silently ignore
  }
}

export function updateStreak(profile: WriterProfile): WriterProfile {
  const { lastActiveDate } = profile.streak;
  if (!lastActiveDate) {
    return { ...profile, streak: { ...profile.streak, current: 0 } };
  }
  const todayStr = today();
  if (lastActiveDate === todayStr) return profile;
  if (isYesterday(lastActiveDate)) {
    const next = profile.streak.current + 1;
    return {
      ...profile,
      streak: {
        ...profile.streak,
        current: next,
        longest: Math.max(next, profile.streak.longest),
      },
    };
  }
  // 2+ days gap — reset
  return { ...profile, streak: { ...profile.streak, current: 0 } };
}

export function markDayActive(profile: WriterProfile): WriterProfile {
  const todayStr = today();
  return {
    ...profile,
    streak: { ...profile.streak, lastActiveDate: todayStr },
  };
}

export function addAnalysisToProfile(
  profile: WriterProfile,
  snapshot: AnalysisSnapshot,
): WriterProfile {
  const newTraitScores = { ...profile.traitScores };
  for (const trait of TRAITS) {
    newTraitScores[trait] = [
      ...profile.traitScores[trait],
      { date: snapshot.date, score: snapshot.traitScores[trait] },
    ];
  }
  return {
    ...profile,
    analysisHistory: [...profile.analysisHistory, snapshot],
    traitScores: newTraitScores,
    stats: {
      ...profile.stats,
      totalAnalyses: profile.stats.totalAnalyses + 1,
    },
  };
}

export function incrementGenreExperience(
  profile: WriterProfile,
  genre: Genre,
): WriterProfile {
  return {
    ...profile,
    genreExperience: {
      ...profile.genreExperience,
      [genre]: profile.genreExperience[genre] + 1,
    },
  };
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function isYesterday(dateStr: string): boolean {
  const d = new Date(dateStr);
  const y = new Date();
  y.setDate(y.getDate() - 1);
  return d.toISOString().slice(0, 10) === y.toISOString().slice(0, 10);
}
