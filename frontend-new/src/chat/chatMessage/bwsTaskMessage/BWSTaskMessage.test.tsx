// mute chatty console
import "src/_test_utilities/consoleMock";

import React from "react";
import { render, screen, fireEvent, waitFor } from "src/_test_utilities/test-utils";
import BWSTaskMessage, { DATA_TEST_ID } from "./BWSTaskMessage";
import { BWSTaskMessageProps } from "./BWSTaskMessage.types";

describe("BWSTaskMessage", () => {
  const getProps = (onSubmit: BWSTaskMessageProps["onSubmit"]): BWSTaskMessageProps => ({
    taskId: "task-1",
    taskNumber: 1,
    totalTasks: 3,
    alternatives: [
      { wa_id: "wa-1", label: "Option one" },
      { wa_id: "wa-2", label: "Option two" },
      { wa_id: "wa-3", label: "Option three" },
    ],
    onSubmit,
  });

  const selectAndSubmit = () => {
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.LETTER_BUTTON("A", "most")));
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.LETTER_BUTTON("B", "least")));
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON));
  };

  test("should stay submitted when the submission succeeds", async () => {
    // GIVEN a submission that succeeds
    const givenOnSubmit = jest.fn().mockResolvedValue(true);
    render(<BWSTaskMessage {...getProps(givenOnSubmit)} />);

    // WHEN the user selects the best and worst options and submits
    selectAndSubmit();

    // THEN expect the selection to have been submitted
    expect(givenOnSubmit).toHaveBeenCalledWith("task-1", "wa-1", "wa-2");
    // AND expect the card to remain disabled
    await waitFor(() => expect(givenOnSubmit).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON)).toBeDisabled();
    expect(screen.getByTestId(DATA_TEST_ID.LETTER_BUTTON("A", "most"))).toBeDisabled();
  });

  test.each([
    ["resolves to false", () => Promise.resolve(false)],
    ["rejects", () => Promise.reject(new Error("failed"))],
  ])("should allow retrying when the submission %s", async (_description, givenSubmitImpl) => {
    // GIVEN a submission that fails the first time and succeeds the second time
    const givenOnSubmit = jest.fn().mockImplementationOnce(givenSubmitImpl).mockResolvedValueOnce(true);
    render(<BWSTaskMessage {...getProps(givenOnSubmit)} />);

    // WHEN the user submits a selection
    selectAndSubmit();

    // THEN expect the card to be re-enabled with the selection kept
    await waitFor(() => expect(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON)).toBeEnabled());
    expect(screen.getByTestId(DATA_TEST_ID.LETTER_BUTTON("A", "most"))).toBeEnabled();

    // AND WHEN the user submits again
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON));

    // THEN expect the same selection to be submitted again
    await waitFor(() => expect(givenOnSubmit).toHaveBeenCalledTimes(2));
    expect(givenOnSubmit).toHaveBeenNthCalledWith(2, "task-1", "wa-1", "wa-2");
    // AND expect the card to be disabled after the successful retry
    await waitFor(() => expect(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON)).toBeDisabled());
  });
});
