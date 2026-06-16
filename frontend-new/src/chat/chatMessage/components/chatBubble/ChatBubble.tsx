import React from "react";
import { ConversationMessageSender } from "src/chat/ChatService/ChatService.types";
import { Box, Link, Typography, styled } from "@mui/material";

// Matches http(s) URLs and www. URLs, stopping at whitespace or common trailing punctuation.
const URL_REGEX = /((?:https?:\/\/|www\.)[^\s<]+[^\s<.,;:!?)\]}'"])/gi;

// Splits a plain-text message into text and clickable link segments, preserving the original text.
const renderTextWithLinks = (text: string): React.ReactNode => {
  const parts = text.split(URL_REGEX);
  return parts.map((part, index) => {
    if (index % 2 === 1) {
      const href = part.startsWith("www.") ? `https://${part}` : part;
      return (
        <Link key={index} href={href} target="_blank" rel="noopener noreferrer">
          {part}
        </Link>
      );
    }
    return part;
  });
};

export interface ChatBubbleProps {
  message: string | React.ReactNode;
  sender: ConversationMessageSender;
  children?: React.ReactNode;
}

const uniqueId = "6e685eeb-2b54-432a-8b66-8a81633b3981";

export const DATA_TEST_ID = {
  CHAT_MESSAGE_BUBBLE_CONTAINER: `chat-message-bubble-container-${uniqueId}`,
  CHAT_MESSAGE_BUBBLE_MESSAGE_TEXT: `chat-message-bubble-message-text-${uniqueId}`,
  CHAT_MESSAGE_BUBBLE_MESSAGE_FOOTER_CONTAINER: `chat-message-bubble-message-footer-container-${uniqueId}`,
};

const MessageBubble = styled(Box)<{ origin: ConversationMessageSender }>(({ theme, origin }) => ({
  width: "fit-content",
  variants: "outlined",
  wordWrap: "break-word",
  wordBreak: "break-word",
  padding: theme.fixedSpacing(theme.tabiyaSpacing.sm),
  border: origin === ConversationMessageSender.USER ? `2px solid ${theme.palette.primary.light}` : "none",
  borderRadius: origin === ConversationMessageSender.USER ? "12px 0px 12px 12px" : "12px 12px 12px 0px",
  backgroundColor:
    origin === ConversationMessageSender.USER
      ? `color-mix(in srgb, ${theme.palette.primary.light} 16%, transparent)`
      : theme.palette.grey[100],
  color: origin === ConversationMessageSender.USER ? theme.palette.primary.contrastText : theme.palette.text.primary,
  position: "relative",
  alignSelf: origin === ConversationMessageSender.USER ? "flex-end" : "flex-start",
  display: "flex",
  flexDirection: "column",
}));

const ChatBubble: React.FC<ChatBubbleProps> = ({ message, sender, children }) => {
  return (
    <MessageBubble origin={sender} data-testid={DATA_TEST_ID.CHAT_MESSAGE_BUBBLE_CONTAINER}>
      <Typography whiteSpace="pre-line" data-testid={DATA_TEST_ID.CHAT_MESSAGE_BUBBLE_MESSAGE_TEXT}>
        {typeof message === "string" ? renderTextWithLinks(message) : message}
      </Typography>
      <Box data-testid={DATA_TEST_ID.CHAT_MESSAGE_BUBBLE_MESSAGE_FOOTER_CONTAINER}>{children}</Box>
    </MessageBubble>
  );
};

export default ChatBubble;
