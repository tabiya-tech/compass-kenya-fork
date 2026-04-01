import "src/_test_utilities/consoleMock";
import ChatService from "./ChatService";
import { StatusCodes } from "http-status-codes";
import { RestAPIError } from "src/error/restAPIError/RestAPIError";
import { setupAPIServiceSpy } from "src/_test_utilities/fetchSpy";
import ErrorConstants from "src/error/restAPIError/RestAPIError.constants";
import { ConversationResponse } from "./ChatService.types";
import { ConversationPhase } from "src/chat/chatProgressbar/types";
import {
  generateTestChatResponses,
  generateTestHistory,
} from "src/chat/ChatService/_test_utilities/generateTestChatResponses";

const serializeSSEEvent = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const serializeCRLFSSEEvent = (event: string, data: unknown) =>
  `event: ${event}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;

describe("ChatService", () => {
  let givenApiServerUrl: string = "/path/to/api";
  beforeEach(() => {
    jest.spyOn(require("src/envService"), "getBackendUrl").mockReturnValue(givenApiServerUrl);
  });
  afterEach(() => {
    jest.clearAllMocks();
  });

  test("should get a single instance successfully", () => {
    // WHEN the service is constructed
    const actualFirstInstance = ChatService.getInstance();

    // THEN expect the service to be constructed successfully
    expect(actualFirstInstance).toBeDefined();

    // AND the service should have the correct endpoint urls
    expect(actualFirstInstance.apiServerUrl).toEqual(givenApiServerUrl);
    expect(actualFirstInstance.chatEndpointUrl).toEqual(`${givenApiServerUrl}/conversations`);

    // AND WHEN the service is constructed again
    const actualSecondInstance = ChatService.getInstance();
    expect(actualFirstInstance).toBe(actualSecondInstance);

    // AND expect no errors or warning to have occurred
    expect(console.error).not.toHaveBeenCalled();
    expect(console.warn).not.toHaveBeenCalled();
  });

  describe("sendMessage", () => {
    test("should fetch the correct URL, with POST and the correct headers and payload successfully", async () => {
      // GIVEN some message specification to send
      const givenMessage = "Hello";
      // AND the send message REST API will respond with OK and some message response
      const expectedMessages = generateTestChatResponses();
      const expectedRootMessageResponse: ConversationResponse = {
        messages: expectedMessages,
        conversation_completed: false,
        conversation_conducted_at: null,
        experiences_explored: 0,
        current_phase: {
          phase: ConversationPhase.INTRO,
          percentage: 0,
          current: null,
          total: null,
        },
      };
      const sseResponseBody = [
        ...expectedMessages.map((message) => serializeSSEEvent("message_completed", message)),
        serializeSSEEvent("turn_completed", {
          conversation_completed: expectedRootMessageResponse.conversation_completed,
          conversation_conducted_at: expectedRootMessageResponse.conversation_conducted_at,
          experiences_explored: expectedRootMessageResponse.experiences_explored,
          current_phase: expectedRootMessageResponse.current_phase,
        }),
      ].join("");
      const fetchSpy = setupAPIServiceSpy(StatusCodes.OK, sseResponseBody, "text/event-stream;charset=UTF-8");

      // WHEN the sendMessage function is called with the given arguments
      const givenSessionId = 1234;
      const service = ChatService.getInstance();
      const actualMessageResponse = await service.sendMessage(givenSessionId, givenMessage);

      // THEN expect it to make a GET request
      // AND the headers
      // AND the request payload to contain the given arguments
      expect(fetchSpy).toHaveBeenCalledWith(`${givenApiServerUrl}/conversations/${givenSessionId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: givenMessage }),
        expectedStatusCode: StatusCodes.OK,
        serviceName: "ChatService",
        serviceFunction: "sendMessage",
        failureMessage: `Failed to send message with session id ${givenSessionId}`,
        expectedContentType: "text/event-stream",
      });

      // AND returns the message response
      expect(actualMessageResponse).toEqual(expectedRootMessageResponse);

      // AND expect no errors or warning to have occurred
      expect(console.error).not.toHaveBeenCalled();
      expect(console.warn).not.toHaveBeenCalled();
    });

    test("on fail to fetch, should reject with the expected service error", async () => {
      const givenMessage = "Hello";
      // GIVEN fetch rejects with some unknown error for sending a message on a given session
      const givenFetchError = new Error("some error");
      jest.spyOn(require("src/utils/customFetch/customFetch"), "customFetch").mockImplementationOnce(() => {
        return new Promise(() => {
          throw givenFetchError;
        });
      });

      // WHEN calling sendMessage function
      const givenSessionId = 1234;
      const service = ChatService.getInstance();

      // THEN expected it to reject with the same error thrown by fetchWithAuth
      await expect(service.sendMessage(givenSessionId, givenMessage)).rejects.toMatchObject(givenFetchError);

      // AND expect no errors or warning to have occurred
      expect(console.error).not.toHaveBeenCalled();
      expect(console.warn).not.toHaveBeenCalled();
    });

    test.each([
      ["is a malformed json", "{"],
      ["is a string", "foo"],
    ])(
      "on 200, should reject with an error ERROR_CODE.INVALID_RESPONSE_BODY if response %s",
      async (_description, givenResponse) => {
        // GIVEN some message specification to send
        const givenMessage = "Hello";
        // AND the send message REST API will respond with an invalid SSE payload
        setupAPIServiceSpy(StatusCodes.OK, givenResponse, "text/event-stream;charset=UTF-8");

        // WHEN the sendMessage function is called with the given arguments
        const givenSessionId = 1234;
        const service = ChatService.getInstance();
        const sendMessagePromise = service.sendMessage(givenSessionId, givenMessage);

        // THEN expected it to reject with the error response
        const expectedError = {
          ...new RestAPIError(
            ChatService.name,
            "sendMessage",
            "POST",
            `${givenApiServerUrl}/conversations`,
            StatusCodes.OK,
            ErrorConstants.ErrorCodes.INVALID_RESPONSE_BODY,
            "",
            ""
          ),
          cause: expect.anything(),
        };
        await expect(sendMessagePromise).rejects.toMatchObject(expectedError);

        // AND expect no errors or warning to have occurred
        expect(console.error).not.toHaveBeenCalled();
        expect(console.warn).not.toHaveBeenCalled();
      }
    );

    test("should dispatch live stream handlers for status, phase, and delta events", async () => {
      const givenMessage = "Hello";
      const expectedMessages = generateTestChatResponses();
      const expectedRootMessageResponse: ConversationResponse = {
        messages: [expectedMessages[0]],
        conversation_completed: false,
        conversation_conducted_at: null,
        experiences_explored: 1,
        current_phase: {
          phase: ConversationPhase.PREFERENCE_ELICITATION,
          percentage: 72,
          current: 1,
          total: 6,
        },
      };
      const sseResponseBody = [
        serializeSSEEvent("turn_started", {
          session_id: 1234,
          user_message_id: "user-1",
          current_phase: {
            phase: ConversationPhase.INTRO,
            percentage: 0,
            current: null,
            total: null,
          },
        }),
        serializeCRLFSSEEvent("status_updated", {
          label: "routing",
          status: "running",
          agent_type: "welcome_agent",
          detail: "INTRO",
          current_phase: {
            phase: ConversationPhase.INTRO,
            percentage: 0,
            current: null,
            total: null,
          },
        }),
        serializeSSEEvent("phase_updated", {
          current_phase: expectedRootMessageResponse.current_phase,
          agent_type: "preference_elicitation_agent",
          detail: "phase_progressed",
        }),
        serializeSSEEvent("message_started", {
          message_id: expectedMessages[0].message_id,
          sender: expectedMessages[0].sender,
          message_type: expectedMessages[0].message_type ?? "TEXT",
          metadata: expectedMessages[0].metadata ?? null,
        }),
        serializeSSEEvent("message_delta", {
          message_id: expectedMessages[0].message_id,
          delta: expectedMessages[0].message.slice(0, 5),
        }),
        serializeSSEEvent("message_completed", expectedMessages[0]),
        serializeSSEEvent("turn_completed", {
          conversation_completed: expectedRootMessageResponse.conversation_completed,
          conversation_conducted_at: expectedRootMessageResponse.conversation_conducted_at,
          experiences_explored: expectedRootMessageResponse.experiences_explored,
          current_phase: expectedRootMessageResponse.current_phase,
        }),
      ].join("");
      setupAPIServiceSpy(StatusCodes.OK, sseResponseBody, "text/event-stream;charset=UTF-8");

      const handlers = {
        onTurnStarted: jest.fn(),
        onStatusUpdated: jest.fn(),
        onPhaseUpdated: jest.fn(),
        onMessageStarted: jest.fn(),
        onMessageDelta: jest.fn(),
        onMessageCompleted: jest.fn(),
        onTurnCompleted: jest.fn(),
      };

      const response = await ChatService.getInstance().sendMessage(1234, givenMessage, handlers);

      expect(response).toEqual(expectedRootMessageResponse);
      expect(handlers.onTurnStarted).toHaveBeenCalledTimes(1);
      expect(handlers.onStatusUpdated).toHaveBeenCalledWith(
        expect.objectContaining({
          label: "routing",
          status: "running",
        })
      );
      expect(handlers.onPhaseUpdated).toHaveBeenCalledWith(
        expect.objectContaining({
          current_phase: expectedRootMessageResponse.current_phase,
        })
      );
      expect(handlers.onMessageStarted).toHaveBeenCalledTimes(1);
      expect(handlers.onMessageDelta).toHaveBeenCalledWith(
        expect.objectContaining({
          message_id: expectedMessages[0].message_id,
          delta: expectedMessages[0].message.slice(0, 5),
        })
      );
      expect(handlers.onMessageCompleted).toHaveBeenCalledWith(expectedMessages[0]);
      expect(handlers.onTurnCompleted).toHaveBeenCalledWith(
        expect.objectContaining({
          current_phase: expectedRootMessageResponse.current_phase,
        })
      );
    });
  });

  describe("getChatHistory", () => {
    test("should fetch the correct URL, with GET and the correct headers and payload successfully", async () => {
      // GIVEN some history to return
      const givenTestHistoryResponse = generateTestHistory();
      const fetchSpy = setupAPIServiceSpy(StatusCodes.OK, givenTestHistoryResponse, "application/json;charset=UTF-8");
      // WHEN the getChatHistory function is called
      const givenSessionId = 1234;
      const service = ChatService.getInstance();
      const actualHistoryResponse = await service.getChatHistory(givenSessionId);

      // THEN expect it to make a GET request
      // AND the headers
      expect(fetchSpy).toHaveBeenCalledWith(`${givenApiServerUrl}/conversations/${givenSessionId}/messages`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        expectedStatusCode: StatusCodes.OK,
        serviceName: "ChatService",
        serviceFunction: "getChatHistory",
        failureMessage: `Failed to get chat history for session id ${givenSessionId}`,
        expectedContentType: "application/json",
        retryOnFailedToFetch: true,
      });

      // AND returns the history response
      expect(actualHistoryResponse).toEqual(givenTestHistoryResponse);

      // AND expect no errors or warning to have occurred
      expect(console.error).not.toHaveBeenCalled();
      expect(console.warn).not.toHaveBeenCalled();
    });

    test("on fail to fetch, should reject with the expected service error", async () => {
      // GIVEN fetch rejects with some unknown error when getting the history of a given session
      const givenFetchError = new Error("some error");
      jest.spyOn(require("src/utils/customFetch/customFetch"), "customFetch").mockImplementationOnce(() => {
        return new Promise(() => {
          throw givenFetchError;
        });
      });

      // WHEN calling getChatHistory function
      const givenSessionId = 1234;
      const service = ChatService.getInstance();

      // THEN expected it to reject with the same error thrown by fetchWithAuth
      await expect(service.getChatHistory(givenSessionId)).rejects.toMatchObject(givenFetchError);

      // AND expect no errors or warning to have occurred
      expect(console.error).not.toHaveBeenCalled();
      expect(console.warn).not.toHaveBeenCalled();
    });

    test.each([
      ["is a malformed json", "{"],
      ["is a string", "foo"],
    ])(
      "on 200, should reject with an error ERROR_CODE.INVALID_RESPONSE_BODY if response %s",
      async (_description, givenResponse) => {
        // GIVEN some message specification to send
        // AND the send message REST API will respond with OK and some response that does conform to the messageResponseSchema even if it states that it is application/json
        setupAPIServiceSpy(StatusCodes.OK, givenResponse, "application/json;charset=UTF-8");

        // WHEN the sendMessage function is called with the given arguments
        const givenSessionId = 1234;
        const service = ChatService.getInstance();
        const sendMessagePromise = service.getChatHistory(givenSessionId);

        // THEN expected it to reject with the error response
        const expectedError = {
          ...new RestAPIError(
            ChatService.name,
            "getChatHistory",
            "GET",
            `${givenApiServerUrl}/conversations`,
            StatusCodes.OK,
            ErrorConstants.ErrorCodes.INVALID_RESPONSE_BODY,
            "",
            ""
          ),
          cause: expect.anything(),
        };
        await expect(sendMessagePromise).rejects.toMatchObject(expectedError);

        // AND expect no errors or warning to have occurred
        expect(console.error).not.toHaveBeenCalled();
        expect(console.warn).not.toHaveBeenCalled();
      }
    );
  });
});
