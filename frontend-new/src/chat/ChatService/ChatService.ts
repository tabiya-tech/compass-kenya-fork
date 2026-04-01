import { getRestAPIErrorFactory } from "src/error/restAPIError/RestAPIError";
import { StatusCodes } from "http-status-codes";
import { customFetch } from "src/utils/customFetch/customFetch";
import ErrorConstants from "src/error/restAPIError/RestAPIError.constants";
import { getBackendUrl } from "src/envService";
import {
  ConversationMessage,
  ConversationResponse,
  ErrorEventData,
  MessageDeltaEventData,
  MessageStartedEventData,
  PhaseUpdatedEventData,
  SendMessageStreamHandlers,
  StatusUpdatedEventData,
  TurnCompletedEventData,
  TurnStartedEventData,
} from "./ChatService.types";

type ParsedSSEEvent = {
  event: string;
  data: unknown;
};

const getSSEBoundary = (buffer: string): { index: number; length: number } | null => {
  const unixBoundary = buffer.indexOf("\n\n");
  const windowsBoundary = buffer.indexOf("\r\n\r\n");
  if (unixBoundary === -1 && windowsBoundary === -1) {
    return null;
  }
  if (unixBoundary === -1) {
    return { index: windowsBoundary, length: 4 };
  }
  if (windowsBoundary === -1) {
    return { index: unixBoundary, length: 2 };
  }
  return unixBoundary < windowsBoundary ? { index: unixBoundary, length: 2 } : { index: windowsBoundary, length: 4 };
};

const parseSSEEvent = (rawEvent: string): ParsedSSEEvent | null => {
  const lines = rawEvent.split(/\r?\n/);
  let event = "message";
  const dataLines: string[] = [];

  lines.forEach((line) => {
    if (!line || line.startsWith(":")) {
      return;
    }
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
      return;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  });

  if (dataLines.length === 0) {
    return null;
  }

  return {
    event,
    data: JSON.parse(dataLines.join("\n")),
  };
};

export default class ChatService {
  private static instance: ChatService;
  readonly chatEndpointUrl: string;
  readonly apiServerUrl: string;
  private constructor() {
    this.apiServerUrl = getBackendUrl();
    this.chatEndpointUrl = `${this.apiServerUrl}/conversations`;
  }

  /**
   * Get the singleton instance of the ChatService.
   * @returns {ChatService} The singleton instance of the ChatService.
   */
  static getInstance(): ChatService {
    if (!ChatService.instance) {
      ChatService.instance = new ChatService();
    }
    return ChatService.instance;
  }

  public async sendMessage(
    sessionId: number,
    message: string,
    handlers?: SendMessageStreamHandlers
  ): Promise<ConversationResponse> {
    const serviceName = "ChatService";
    const serviceFunction = "sendMessage";
    const method = "POST";
    const errorFactory = getRestAPIErrorFactory(serviceName, serviceFunction, method, this.chatEndpointUrl);
    const constructedSendMessageURL = `${this.chatEndpointUrl}/${sessionId}/messages`;

    const response = await customFetch(constructedSendMessageURL, {
      method: method,
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_input: message,
      }),
      expectedStatusCode: StatusCodes.OK,
      serviceName,
      serviceFunction,
      failureMessage: `Failed to send message with session id ${sessionId}`,
      expectedContentType: "text/event-stream",
    });

    const completedMessages: ConversationMessage[] = [];
    let completedTurn: TurnCompletedEventData | null = null;
    let streamError: ErrorEventData | null = null;

    const processRawEvent = (rawEvent: string) => {
      if (!rawEvent.trim()) {
        return;
      }
      let parsedEvent: ParsedSSEEvent | null = null;
      try {
        parsedEvent = parseSSEEvent(rawEvent);
      } catch (e: any) {
        throw errorFactory(
          response.status,
          ErrorConstants.ErrorCodes.INVALID_RESPONSE_BODY,
          "Response did not contain valid SSE JSON payloads",
          {
            rawEvent,
            error: e,
          }
        );
      }
      if (!parsedEvent) {
        return;
      }

      switch (parsedEvent.event) {
        case "turn_started":
          handlers?.onTurnStarted?.(parsedEvent.data as TurnStartedEventData);
          break;
        case "status_updated":
          handlers?.onStatusUpdated?.(parsedEvent.data as StatusUpdatedEventData);
          break;
        case "phase_updated":
          handlers?.onPhaseUpdated?.(parsedEvent.data as PhaseUpdatedEventData);
          break;
        case "message_started":
          handlers?.onMessageStarted?.(parsedEvent.data as MessageStartedEventData);
          break;
        case "message_delta":
          handlers?.onMessageDelta?.(parsedEvent.data as MessageDeltaEventData);
          break;
        case "message_completed": {
          const completedMessage = parsedEvent.data as ConversationMessage;
          completedMessages.push(completedMessage);
          handlers?.onMessageCompleted?.(completedMessage);
          break;
        }
        case "turn_completed":
          completedTurn = parsedEvent.data as TurnCompletedEventData;
          handlers?.onTurnCompleted?.(completedTurn);
          break;
        case "error":
          streamError = parsedEvent.data as ErrorEventData;
          handlers?.onError?.(streamError);
          break;
        default:
          break;
      }
    };

    const processBuffer = (buffer: string) => {
      let boundary = getSSEBoundary(buffer);
      while (boundary) {
        processRawEvent(buffer.slice(0, boundary.index));
        buffer = buffer.slice(boundary.index + boundary.length);
        boundary = getSSEBoundary(buffer);
      }
      if (buffer.trim()) {
        processRawEvent(buffer);
      }
    };

    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const boundary = getSSEBoundary(buffer);
        if (boundary) {
          processRawEvent(buffer.slice(0, boundary.index));
          buffer = buffer.slice(boundary.index + boundary.length);
        }
        if (done) {
          processBuffer(buffer);
          break;
        }
      }
    } else {
      const text = await response.text();
      processBuffer(text);
    }

    if (streamError) {
      const errorEvent = streamError as ErrorEventData;
      throw errorFactory(response.status, ErrorConstants.ErrorCodes.API_ERROR, errorEvent.message, errorEvent);
    }

    if (!completedTurn) {
      throw errorFactory(
        response.status,
        ErrorConstants.ErrorCodes.INVALID_RESPONSE_BODY,
        "SSE stream ended without turn completion",
        {
          messagesReceived: completedMessages.length,
        }
      );
    }

    const completedTurnData = completedTurn as TurnCompletedEventData;
    return {
      messages: completedMessages,
      conversation_completed: completedTurnData.conversation_completed,
      conversation_conducted_at: completedTurnData.conversation_conducted_at,
      experiences_explored: completedTurnData.experiences_explored,
      current_phase: completedTurnData.current_phase,
    };
  }

  public async getChatHistory(sessionId: number): Promise<ConversationResponse> {
    const serviceName = "ChatService";
    const serviceFunction = "getChatHistory";
    const method = "GET";
    const constructedHistoryURL = `${this.chatEndpointUrl}/${sessionId}/messages`;

    const response = await customFetch(constructedHistoryURL, {
      method: method,
      headers: { "Content-Type": "application/json" },
      expectedStatusCode: StatusCodes.OK,
      serviceName,
      serviceFunction,
      failureMessage: `Failed to get chat history for session id ${sessionId}`,
      expectedContentType: "application/json",
      retryOnFailedToFetch: true,
    });

    const responseBody = await response.text();

    let chatHistory: ConversationResponse;
    try {
      chatHistory = JSON.parse(responseBody);
    } catch (e: any) {
      const errorFactory = getRestAPIErrorFactory(serviceName, serviceFunction, method, this.chatEndpointUrl);
      throw errorFactory(
        response.status,
        ErrorConstants.ErrorCodes.INVALID_RESPONSE_BODY,
        "Response did not contain valid JSON",
        {
          responseBody,
          error: e,
        }
      );
    }

    return chatHistory;
  }
}
