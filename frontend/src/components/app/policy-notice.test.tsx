import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PolicyNotice } from "./policy-notice";

describe("PolicyNotice", () => {
  it("renders nothing for an allowed decision", () => {
    const { container } = render(
      <PolicyNotice notice={{ id: "p", decision: "allowed", reason: "Fine" }} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the backend's own explanation for a denial", () => {
    render(
      <PolicyNotice
        notice={{ id: "p", decision: "denied", reason: "Exporting contacts isn't allowed." }}
      />,
    );
    expect(screen.getByText("This action wasn't permitted")).toBeInTheDocument();
    expect(screen.getByText("Exporting contacts isn't allowed.")).toBeInTheDocument();
  });

  it("offers approval controls only when the backend marks the action approvable", () => {
    const { rerender } = render(
      <PolicyNotice
        notice={{ id: "p", decision: "requires_approval", reason: "Confirm this step." }}
      />,
    );
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();

    rerender(
      <PolicyNotice
        notice={{
          id: "p",
          decision: "requires_approval",
          reason: "Confirm this step.",
          approvable: true,
        }}
      />,
    );
    expect(screen.getByRole("button", { name: "Approve" })).toBeInTheDocument();
  });
});
