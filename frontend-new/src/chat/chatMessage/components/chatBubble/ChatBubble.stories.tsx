import type { Meta, StoryObj } from "@storybook/react";
import ChatBubble from "./ChatBubble";
import { ConversationMessageSender } from "src/chat/ChatService/ChatService.types";
import { VisualMock } from "src/_test_utilities/VisualMock";

const meta: Meta<typeof ChatBubble> = {
  title: "Chat/ChatBubble",
  component: ChatBubble,
  tags: ["autodocs"],
  argTypes: {},
};

export default meta;

type Story = StoryObj<typeof ChatBubble>;

export const FromCompass: Story = {
  args: {
    message: "Hello, how can I help you?",
    sender: ConversationMessageSender.COMPASS,
  },
};

export const FromUser: Story = {
  args: {
    message: "Hi there, I am a baker!",
    sender: ConversationMessageSender.USER,
  },
};

export const ShownWithFooter: Story = {
  args: {
    message: "Hello, how can I help you?",
    sender: ConversationMessageSender.COMPASS,
    children: <VisualMock text={"Foo Footer"} />,
  },
};

export const WithClickableLinks: Story = {
  args: {
    message:
      "Here are some opportunities that match you:\n\n" +
      "1. Junior Baker\n" +
      "   Employer: Sunrise Bakery\n" +
      "   Apply here: https://jobs.example.com/junior-baker\n\n" +
      "2. Pastry Assistant\n" +
      "   Employer: City Cafe\n" +
      "   Apply here: www.citycafe.co.ke/careers/pastry-assistant",
    sender: ConversationMessageSender.COMPASS,
  },
};
