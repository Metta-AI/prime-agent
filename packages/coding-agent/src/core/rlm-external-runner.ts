/**
 * External rlm() runner — the seam that lets a substrate own subagents.
 *
 * When `settings.rlmRunCommand` (or PRIME_AGENT_RLM_RUN_COMMAND) names a
 * command, an admitted `rlm()` child is not an in-process AgentSession: the
 * command is spawned once per child and owns the whole run. Protocol:
 *
 * - stdin: one JSON object, the {@link ExternalRlmRunRequest} (then EOF);
 * - stdout: JSON lines, {@link ExternalRlmRunEvent}s — `status` lines
 *   update the child's live activity, `done` carries the child's answer
 *   (delivered to the parent as an agent message from the child), `error`
 *   fails the child; a non-JSON line is a `status`;
 * - exit: 0 after `done`; {@link EXTERNAL_RLM_RUNNER_DECLINED_EXIT_CODE}
 *   BEFORE any event means "not mine" and the child runs in-process as if
 *   no runner were configured; anything else is a failure whose message is
 *   the runner's stderr tail.
 *
 * The runner is trusted exactly like `shellPath`: it is configuration the
 * user installed.
 */

import { spawn } from "node:child_process";

export const EXTERNAL_RLM_RUNNER_DECLINED_EXIT_CODE = 75;

export interface ExternalRlmRunRequest {
	protocol: 1;
	prompt: string;
	childId: string;
	sessionName: string;
	sessionDir: string;
	model: string;
	parentSessionId: string;
	parentSessionName?: string;
	parentSessionDir?: string;
	cwd: string;
	rlmDepth: number;
	rlmMaxDepth: number;
	spawnCode?: string;
}

export type ExternalRlmRunEvent =
	| { event: "status"; text: string }
	| { event: "done"; answer: string; detail?: Record<string, unknown> }
	| { event: "error"; error: string };

export interface ExternalRlmRunOutcome {
	answer: string;
	detail?: Record<string, unknown>;
}

export interface ExternalRlmChildHandle {
	/** Resolves true once the runner has claimed the child (any event, or a non-declined exit); false when it declined. */
	admitted: Promise<boolean>;
	/** The child's outcome; rejects on runner failure or abort. Never settles before `admitted`. */
	result: Promise<ExternalRlmRunOutcome>;
	abort: (reason?: string) => void;
}

const STDERR_TAIL_CHARS = 4000;

function parseEvent(line: string): ExternalRlmRunEvent | undefined {
	const trimmed = line.trim();
	if (!trimmed) return undefined;
	if (!trimmed.startsWith("{")) return { event: "status", text: trimmed };
	try {
		const parsed = JSON.parse(trimmed) as Record<string, unknown>;
		switch (parsed.event) {
			case "status":
				return { event: "status", text: String(parsed.text ?? "") };
			case "done":
				return {
					event: "done",
					answer: typeof parsed.answer === "string" ? parsed.answer : "",
					detail:
						typeof parsed.detail === "object" && parsed.detail !== null
							? (parsed.detail as Record<string, unknown>)
							: undefined,
				};
			case "error":
				return { event: "error", error: String(parsed.error ?? "external rlm runner reported an error") };
			default:
				return { event: "status", text: trimmed };
		}
	} catch {
		return { event: "status", text: trimmed };
	}
}

/** Spawn the runner for one child and drive its event stream. */
export function startExternalRlmChild(
	command: string,
	request: ExternalRlmRunRequest,
	options: { onEvent?: (event: ExternalRlmRunEvent) => void } = {},
): ExternalRlmChildHandle {
	let resolveAdmitted!: (value: boolean) => void;
	const admitted = new Promise<boolean>((resolve) => {
		resolveAdmitted = resolve;
	});
	let resolveResult!: (value: ExternalRlmRunOutcome) => void;
	let rejectResult!: (error: Error) => void;
	const result = new Promise<ExternalRlmRunOutcome>((resolve, reject) => {
		resolveResult = resolve;
		rejectResult = reject;
	});
	// A rejection nobody awaits yet (a declined run) must not surface as unhandled.
	result.catch(() => undefined);

	let claimed = false;
	let settled = false;
	let outcome: ExternalRlmRunOutcome | undefined;
	let reportedError: string | undefined;
	let abortReason: string | undefined;
	let stderrTail = "";

	const claim = () => {
		if (!claimed) {
			claimed = true;
			resolveAdmitted(true);
		}
	};
	const finish = (error?: Error, wasDeclined = false) => {
		if (settled) return;
		settled = true;
		if (!claimed) {
			claimed = true;
			resolveAdmitted(!wasDeclined);
		}
		if (error) rejectResult(error);
		else if (outcome) resolveResult(outcome);
		else rejectResult(new Error("external rlm runner exited without a result"));
	};

	const child = spawn(command, [], {
		cwd: request.cwd,
		stdio: ["pipe", "pipe", "pipe"],
		env: {
			...process.env,
			PRIME_AGENT_RLM_CHILD_ID: request.childId,
			RLM_DEPTH: String(request.rlmDepth),
			RLM_MAX_DEPTH: String(request.rlmMaxDepth),
		},
	});

	const handleEvent = (event: ExternalRlmRunEvent) => {
		claim();
		if (event.event === "done") {
			outcome = { answer: event.answer, detail: event.detail };
		} else if (event.event === "error") {
			reportedError = event.error;
		}
		try {
			options.onEvent?.(event);
		} catch {
			// A display hook must never take the child down.
		}
	};

	let stdoutBuffer = "";
	child.stdout.setEncoding("utf8");
	child.stdout.on("data", (chunk: string) => {
		stdoutBuffer += chunk;
		let newline = stdoutBuffer.indexOf("\n");
		while (newline >= 0) {
			const line = stdoutBuffer.slice(0, newline);
			stdoutBuffer = stdoutBuffer.slice(newline + 1);
			const event = parseEvent(line);
			if (event) handleEvent(event);
			newline = stdoutBuffer.indexOf("\n");
		}
	});
	child.stderr.setEncoding("utf8");
	child.stderr.on("data", (chunk: string) => {
		stderrTail = (stderrTail + chunk).slice(-STDERR_TAIL_CHARS);
	});
	child.on("error", (error) => {
		finish(new Error(`external rlm runner ${command}: ${error.message}`));
	});
	child.on("close", (code, signal) => {
		const trailing = parseEvent(stdoutBuffer);
		if (trailing) handleEvent(trailing);
		if (abortReason !== undefined) {
			finish(new Error(abortReason));
			return;
		}
		if (code === EXTERNAL_RLM_RUNNER_DECLINED_EXIT_CODE && !claimed) {
			finish(new Error("external rlm runner declined"), true);
			return;
		}
		if (reportedError !== undefined) {
			finish(new Error(reportedError));
			return;
		}
		if (code !== 0) {
			const why = signal ? `killed by ${signal}` : `exit ${code}`;
			const tail = stderrTail.trim();
			finish(new Error(`external rlm runner failed (${why})${tail ? `: ${tail}` : ""}`));
			return;
		}
		if (!outcome) {
			finish(new Error("external rlm runner exited 0 without a done event"));
			return;
		}
		finish();
	});
	child.stdin.on("error", () => undefined);
	child.stdin.end(`${JSON.stringify(request)}\n`);

	return {
		admitted,
		result,
		abort: (reason?: string) => {
			if (settled) return;
			abortReason = reason ?? "RLM child cancelled";
			claim();
			try {
				child.kill("SIGTERM");
			} catch {
				// Already gone.
			}
			setTimeout(() => {
				if (!settled) {
					try {
						child.kill("SIGKILL");
					} catch {
						// Already gone.
					}
				}
			}, 5000).unref();
		},
	};
}
