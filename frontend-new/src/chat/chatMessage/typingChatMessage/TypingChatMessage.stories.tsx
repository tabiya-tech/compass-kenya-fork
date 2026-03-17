import TypingChatMessage, { WAIT_BEFORE_THINKING } from "src/chat/chatMessage/typingChatMessage/TypingChatMessage";
import { Meta, StoryObj } from "@storybook/react";

const meta: Meta<typeof TypingChatMessage> = {
  title: "Chat/ChatMessage/TypingChatMessage",
  component: TypingChatMessage,
  tags: ["autodocs"],
};

export default meta;

type Story = StoryObj<typeof TypingChatMessage>;

export const Shown: Story = {
  args: {},
};

export const ShownWithQuickTyping: Story = {
  args: {
    waitBeforeThinking: 1000,
  },
};

export const ShownWithTyping: Story = {
  args: {
    waitBeforeThinking: WAIT_BEFORE_THINKING,
  },
};

export const ShownWhenThinking: Story = {
  args: {
    waitBeforeThinking: 0,
  },
};

export const WithStreamingStatusPreparing: Story = {
  args: {
    waitBeforeThinking: 0,
    status: "Preparing your response",
  },
};

export const WithStreamingStatusExploring: Story = {
  args: {
    waitBeforeThinking: 0,
    status: "Exploring your experiences",
  },
};

export const WithStreamingStatusUnderstandingPreferences: Story = {
  args: {
    waitBeforeThinking: 0,
    status: "Understanding what matters most to you",
  },
};

export const WithStreamingStatusPreparingRecommendations: Story = {
  args: {
    waitBeforeThinking: 0,
    status: "Preparing your recommendations",
  },
};
