// Enum for the sender
import { ReactionKind } from "src/chat/reaction/reaction.types";
import { CurrentPhase } from "src/chat/chatProgressbar/types";
import { BWSTaskMetadata } from "src/chat/chatMessage/bwsTaskMessage/BWSTaskMessage.types";

export enum ConversationMessageSender {
  USER = "USER",
  COMPASS = "COMPASS",
}
export interface MessageReaction {
  id: string;
  kind: ReactionKind | null;
}

export type ConversationMessageType = "TEXT" | "BWS_TASK";

// Type for individual conversation messages
export interface ConversationMessage {
  message_id: string;
  message: string;
  sent_at: string; // ISO formatted datetime string
  sender: ConversationMessageSender; // Either 'USER' or 'COMPASS'
  reaction: MessageReaction | null;
  message_type?: ConversationMessageType;
  metadata?: BWSTaskMetadata;
}

export interface ConversationResponse {
  messages: ConversationMessage[];
  conversation_completed: boolean;
  conversation_conducted_at: string | null; // ISO formatted datetime string
  experiences_explored: number; // a count for all the experiences explored (processed)
  current_phase: CurrentPhase; // The current conversation phase
}
