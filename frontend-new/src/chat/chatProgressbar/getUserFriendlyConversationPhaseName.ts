import { ConversationPhase, CurrentPhase } from "./types";
import i18n from "src/i18n/i18n";

// Use a Proxy to create a getter pattern that fetches translations on-demand.
// This ensures that if the user changes the language, translations are always up-to-date.
const getUserFriendlyPhaseNames = (): Record<ConversationPhase, string> => {
  return new Proxy({} as Record<ConversationPhase, string>, {
    get(target, prop: string | symbol) {
      if (typeof prop !== "string") return undefined;

      const phase = prop as ConversationPhase;
      const translationMap: Record<ConversationPhase, string> = {
        [ConversationPhase.INITIALIZING]: "chat.chatProgressbar.phases.initializing",
        [ConversationPhase.INTRO]: "chat.chatProgressbar.phases.intro",
        [ConversationPhase.COLLECT_EXPERIENCES]: "chat.chatProgressbar.phases.collecting",
        [ConversationPhase.DIVE_IN]: "chat.chatProgressbar.phases.exploring",
        [ConversationPhase.PREFERENCE_ELICITATION]: "chat.chatProgressbar.phases.discoveringPreferences",
        [ConversationPhase.ENDED]: "chat.chatProgressbar.phases.finished",
        [ConversationPhase.UNKNOWN]: "chat.chatProgressbar.phases.unknown",
      };

      return i18n.t(translationMap[phase] || translationMap[ConversationPhase.UNKNOWN]);
    },
  });
};

/**
 * Get a user-friendly name for the conversation phase
 * @param phase {CurrentPhase} - The conversation phase object.
 * @returns {string} User-friendly, translated name (with progress if applicable)
 */
export function getUserFriendlyConversationPhaseName(phase: CurrentPhase): string {
  const USER_FRIENDLY_PHASE_NAMES = getUserFriendlyPhaseNames();
  if (phase.phase === ConversationPhase.COLLECT_EXPERIENCES) {
    return `${USER_FRIENDLY_PHASE_NAMES[ConversationPhase.COLLECT_EXPERIENCES]}: ${phase.current}/${phase.total} ${i18n.t("chat.chatProgressbar.labels.workTypes")}`;
  }

  if (phase.phase === ConversationPhase.DIVE_IN) {
    return `${USER_FRIENDLY_PHASE_NAMES[ConversationPhase.DIVE_IN]}: ${phase.current}/${phase.total} ${i18n.t("chat.chatProgressbar.labels.experiences")}`;
  }

  if (phase.phase === ConversationPhase.PREFERENCE_ELICITATION) {
    // Show progress if current and total are provided
    if (phase.current !== null && phase.total !== null) {
      // BWS phase has total=12 (occupation ranking tasks)
      // Use different label for BWS vs regular questions
      const label =
        phase.total === 12
          ? i18n.t("chat.chatProgressbar.labels.tasks")
          : i18n.t("chat.chatProgressbar.labels.questions");

      const phaseName =
        phase.total === 12
          ? i18n.t("chat.chatProgressbar.phases.rankingOccupations")
          : USER_FRIENDLY_PHASE_NAMES[ConversationPhase.PREFERENCE_ELICITATION];

      return `${phaseName}: ${phase.current}/${phase.total} ${label}`;
    }
    // Otherwise just show the phase name
    return USER_FRIENDLY_PHASE_NAMES[ConversationPhase.PREFERENCE_ELICITATION];
  }

  return USER_FRIENDLY_PHASE_NAMES[phase.phase] || USER_FRIENDLY_PHASE_NAMES[ConversationPhase.UNKNOWN];
}
