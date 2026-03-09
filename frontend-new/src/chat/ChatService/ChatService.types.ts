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

export type ConversationStreamEventType =
  | "turn_started"
  | "message_started"
  | "message_delta"
  | "message_completed"
  | "turn_completed"
  | "error";

export interface TurnStartedEventData {
  session_id: number;
  user_message_id: string;
  current_phase: CurrentPhase | null;
}

export interface MessageStartedEventData {
  message_id: string;
  sender: ConversationMessageSender;
  message_type?: ConversationMessageType;
  metadata?: BWSTaskMetadata;
}

export interface MessageDeltaEventData {
  message_id: string;
  delta: string;
}

export interface TurnCompletedEventData {
  conversation_completed: boolean;
  conversation_conducted_at: string | null;
  experiences_explored: number;
  current_phase: CurrentPhase;
}

export interface ErrorEventData {
  code: string;
  message: string;
  recoverable: boolean;
}

export interface SendMessageStreamHandlers {
  onTurnStarted?: (event: TurnStartedEventData) => void;
  onMessageStarted?: (event: MessageStartedEventData) => void;
  onMessageDelta?: (event: MessageDeltaEventData) => void;
  onMessageCompleted?: (event: ConversationMessage) => void;
  onTurnCompleted?: (event: TurnCompletedEventData) => void;
  onError?: (event: ErrorEventData) => void;
}
