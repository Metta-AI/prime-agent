/**
 * Regression tests: the agent must keep working after an auto-compaction that
 * interrupted unfinished work ("agent sometimes does not continue after
 * auto-compaction").
 *
 * Covered bugs:
 *
 * BUG A: A threshold auto-compaction that intentionally stopped a mid-task tool loop
 *   (shouldStopAfterTurn -> true with _continueAfterThresholdCompaction=true) must
 *   resume the loop even when the compaction itself fails or is skipped
 *   (resumeAfterFailure in agent-session.ts used to fire for reason==="requested" only).
 *
 * BUG B: With an active goal, a threshold compaction whose stop lands at the end of an
 *   assistant TEXT turn used to set _continueAfterThresholdCompaction=false via the
 *   role-based heuristic (lastMessage.role !== "assistant") and, after a SUCCESSFUL
 *   compaction, nothing restarted the goal-continuation loop; the goal now queues its
 *   continuation as a session input before compaction.
 *
 * A control test pins the pre-existing requested-compaction resume behavior.
 */
import type { AgentMessage, ShouldStopAfterTurnContext } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	fauxAssistantMessage,
	fauxToolCall,
	type ToolResultMessage,
	type Usage,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentSession } from "../../src/core/agent-session.js";
import { createHarness, type Harness } from "./harness.js";

type SessionInternals = {
	_shouldStopAfterTurn: (context: ShouldStopAfterTurnContext) => boolean | Promise<boolean>;
	_runAutoCompaction: (reason: "overflow" | "threshold" | "requested", willRetry: boolean) => Promise<boolean>;
	_continueAfterThresholdCompaction: boolean;
};

function createUsage(totalTokens: number): Usage {
	return {
		input: totalTokens,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createAssistant(
	harness: Harness,
	options: { stopReason?: AssistantMessage["stopReason"]; totalTokens?: number; timestamp?: number },
): AssistantMessage {
	const model = harness.getModel();
	return {
		...fauxAssistantMessage("", { stopReason: options.stopReason, timestamp: options.timestamp }),
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: createUsage(options.totalTokens ?? 0),
	};
}

/** Faux ipython tool that services goal.* host requests like the real kernel bridge. */
function createFauxIpythonTool(sessionRef: { current?: AgentSession }) {
	return {
		name: "ipython",
		label: "ipython",
		description: "Execute Python code in the agent kernel.",
		parameters: Type.Object({ code: Type.String() }),
		execute: async (_toolCallId: string, params: unknown) => {
			const session = sessionRef.current;
			if (!session) throw new Error("test session is not initialized");
			const code = (params as { code: string }).code.trim();
			let text = "";
			if (code.startsWith("goal.")) {
				const spaceIndex = code.indexOf(" ");
				const type = spaceIndex < 0 ? code : code.slice(0, spaceIndex);
				const payload = spaceIndex < 0 ? {} : JSON.parse(code.slice(spaceIndex + 1));
				text = JSON.stringify(session.handleGoalHostRequest(type, payload));
			}
			return { content: [{ type: "text" as const, text }], details: {} };
		},
	};
}

describe("compaction continuation", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	function midToolLoopContext(harness: Harness): ShouldStopAfterTurnContext {
		const assistant = createAssistant(harness, {
			stopReason: "toolUse",
			totalTokens: 250_000,
			timestamp: Date.now(),
		});
		const toolResult: ToolResultMessage<unknown> = {
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "big",
			content: [{ type: "text", text: "result" }],
			isError: false,
			timestamp: Date.now() + 500,
		};
		const messages: AgentMessage[] = [
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() - 1000 },
			assistant,
			toolResult,
		];
		harness.session.agent.state.messages = messages;
		return {
			message: assistant,
			toolResults: [toolResult],
			context: { systemPrompt: harness.session.systemPrompt, messages, tools: [] },
			newMessages: [assistant, toolResult],
		};
	}

	// BUG A (unit-level).
	it("resumes the interrupted tool loop when a threshold compaction is skipped", async () => {
		vi.useFakeTimers();
		const harness = await createHarness({
			settings: { compaction: { enabled: true, reserveTokens: 1000 } },
			models: [{ id: "faux-1", contextWindow: 200_000 }],
		});
		harnesses.push(harness);
		const internals = harness.session as unknown as SessionInternals;
		const context = midToolLoopContext(harness);

		// Mid-tool-loop: last message is a toolResult, so the session decides to
		// stop the loop for compaction AND continue afterwards.
		const shouldStop = await internals._shouldStopAfterTurn(context);
		expect(shouldStop).toBe(true);
		expect(internals._continueAfterThresholdCompaction).toBe(true);

		const continueSpy = vi.spyOn(harness.session.agent, "continue").mockResolvedValue();

		// The in-memory session has no persisted entries, so _performCompaction throws
		// CompactionSkippedError — standing in for any summarizer failure.
		await internals._runAutoCompaction("threshold", false);
		await vi.advanceTimersByTimeAsync(500);

		const endEvents = harness.eventsOfType("compaction_end");
		expect(endEvents).toHaveLength(1);
		expect(endEvents[0].errorMessage).toContain("skipped");

		// The loop that compaction interrupted resumes even though the compaction
		// itself was skipped/failed (regression: resumeAfterFailure used to fire
		// for "requested" only, so continue() was never called).
		expect(continueSpy).toHaveBeenCalledTimes(1);
	});

	// CONTROL (passes today): the identical failure for reason === "requested" DOES
	// resume, proving the asymmetry at agent-session.ts ~8185-8192.
	it("control: a skipped requested compaction mid tool loop does resume", async () => {
		vi.useFakeTimers();
		const harness = await createHarness({
			settings: { compaction: { enabled: true, reserveTokens: 1000 } },
			models: [{ id: "faux-1", contextWindow: 200_000 }],
		});
		harnesses.push(harness);
		const internals = harness.session as unknown as SessionInternals;
		midToolLoopContext(harness);
		internals._continueAfterThresholdCompaction = true;

		const continueSpy = vi.spyOn(harness.session.agent, "continue").mockResolvedValue();

		await internals._runAutoCompaction("requested", false);
		await vi.advanceTimersByTimeAsync(500);

		expect(continueSpy).toHaveBeenCalledTimes(1);
	});

	// BUG A (end-to-end). A real prompt-driven tool loop is
	// interrupted for threshold compaction; the compaction is skipped; the queued
	// final model response is never consumed and the session goes idle mid-task.
	it("e2e: tool loop interrupted by a skipped threshold compaction resumes", async () => {
		const bigTool = {
			name: "big",
			label: "big",
			description: "returns big text",
			parameters: Type.Object({}),
			execute: async () => ({
				content: [{ type: "text" as const, text: "x".repeat(40_000) }],
				details: {},
			}),
		};
		const harness = await createHarness({
			tools: [bigTool],
			// Huge keepRecentTokens: prepareCompaction finds nothing to summarize and
			// throws CompactionSkippedError, standing in for a failed summarizer call.
			settings: { compaction: { enabled: true, reserveTokens: 500, keepRecentTokens: 1_000_000 } },
			models: [{ id: "faux-1", contextWindow: 6_000 }],
			persistSession: true,
		});
		harnesses.push(harness);
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("big", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("final answer after the tool call"),
		]);

		await harness.session.prompt("run the tool then summarize");
		await new Promise((resolve) => setTimeout(resolve, 300));
		await harness.session.waitForIdle();
		await new Promise((resolve) => setTimeout(resolve, 300));

		// The threshold compaction fired mid tool loop and was skipped.
		expect(harness.eventsOfType("compaction_start").map((event) => event.reason)).toContain("threshold");
		expect(harness.eventsOfType("compaction_end")[0]?.errorMessage).toContain("skipped");

		// The interrupted loop resumes and consumes the final response
		// (regression: it stayed pending and the session sat idle mid-task).
		expect(harness.getPendingResponseCount()).toBe(0);
	});

	// BUG B (end-to-end). With an active goal, a SUCCESSFUL
	// threshold compaction at the end of an assistant text turn stops the goal loop.
	it("e2e: an active goal keeps continuing after a successful threshold compaction", async () => {
		const sessionRef: { current?: AgentSession } = {};
		const harness = await createHarness({
			tools: [createFauxIpythonTool(sessionRef)],
			// Small window; faux-provider usage grows a few hundred tokens per turn,
			// so a later goal-continuation turn crosses the threshold.
			settings: { compaction: { enabled: true, reserveTokens: 500, keepRecentTokens: 1 } },
			models: [{ id: "faux-1", contextWindow: 4_300 }],
			persistSession: true,
			extensionFactories: [
				(pi) => {
					pi.on("session_before_compact", async (event) => ({
						compaction: {
							summary: "auto compacted",
							firstKeptEntryId: event.preparation.firstKeptEntryId,
							tokensBefore: event.preparation.tokensBefore,
							details: {},
						},
					}));
				},
			],
		});
		harnesses.push(harness);
		sessionRef.current = harness.session;
		harness.setResponses([
			fauxAssistantMessage("step one done, more to do"),
			fauxAssistantMessage("step two done, still more to do"),
			fauxAssistantMessage("step three done, still not finished"),
			fauxAssistantMessage(fauxToolCall("ipython", { code: "goal.complete" }), { stopReason: "toolUse" }),
			fauxAssistantMessage("Goal complete."),
		]);

		await harness.session.prompt("/goal finish the task");
		await new Promise((resolve) => setTimeout(resolve, 300));
		await harness.session.waitForIdle();
		await new Promise((resolve) => setTimeout(resolve, 300));

		// A threshold compaction did happen mid-goal and SUCCEEDED.
		expect(harness.eventsOfType("compaction_start").map((event) => event.reason)).toContain("threshold");
		expect(harness.eventsOfType("compaction_end")[0]?.result).toBeDefined();

		// The active goal keeps driving turns after compaction and completes
		// (regression: the goal stayed "active" forever and the agent stopped).
		expect(harness.getPendingResponseCount()).toBe(0);
		expect(harness.session.goalState.status).toBe("complete");
	});

	// Goal and autonomous mode can be active at the same time. At a threshold
	// boundary the goal continuation must take exclusive priority, matching
	// _getContinuationMessages — not queue one continuation per driver.
	it("queues only the goal continuation when a goal and autonomous mode are both active", async () => {
		const sessionRef: { current?: AgentSession } = {};
		const harness = await createHarness({
			tools: [createFauxIpythonTool(sessionRef)],
			autonomous: { enabled: true, maxContinuations: 5 },
			settings: { compaction: { enabled: true, reserveTokens: 1000 } },
			models: [{ id: "faux-1", contextWindow: 200_000 }],
		});
		harnesses.push(harness);
		sessionRef.current = harness.session;
		harness.session.handleGoalHostRequest("goal.create", { objective: "finish the task" });
		const internals = harness.session as unknown as SessionInternals;
		const context = midToolLoopContext(harness);

		const shouldStop = await internals._shouldStopAfterTurn(context);
		expect(shouldStop).toBe(true);
		expect(internals._continueAfterThresholdCompaction).toBe(true);

		// Exactly one continuation is queued, and it is the goal's.
		expect(harness.session.queuedActionCount).toBe(1);
		expect(harness.session.goalState.continuationsUsed).toBe(1);
		expect(harness.session.getAutonomousStatus().continuationsUsed).toBe(0);
	});
});
