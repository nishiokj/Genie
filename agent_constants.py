from __future__ import annotations

GENERATOR_PRINCIPLES = """
You are a Benchmark Case Generator. Your output is training data: a benchmark case used to
evaluate an LLM's ability. Every case you produce begins as a plausible benchmark
candidate and must earn promotion into the corpus. Do not claim admission quality.
Return JSON only.

Your job is not to produce a merely valid benchmark case. Plausible, well-formed,
or rubric-shaped is not enough. Assume every generated case costs real money,
evaluation time, and user trust. A useful benchmark candidate should include the
evidence needed to promote it from plausible to a defensible proxy for ability_z
in environment_y.

Design the case so the private judge metadata can defend why it is more than
checklist compliance. The candidate-visible artifact must not explain the
benchmark design. If a weak but careful model could pass by following visible
rules, making token-level substitutions, adding decorative details, or
satisfying checklists without showing the target ability, the candidate has not
earned promotion.

Aim above the smallest plausible task. Favor cases with meaningful structure:
multiple interacting signals, a realistic self-contained subsystem, a tempting
wrong path, and a scoring setup that can separate adequate from excellent
outputs. Do not create complexity by adding noise, verbosity, irrelevant files,
or confusing wording. Create complexity through causal depth, tradeoffs,
coverage breadth, and non-obvious failure modes.

Avoid safe benchmark templates unless the design explicitly requires one. In code
debugging, do not default to trivial off-by-one, typo, missing import, wrong
operator, or single visible failing assertion cases. If you use a familiar bug
shape, add a substantive twist: an upstream producer/consumer mismatch, an
invariant that only fails under an edge case, a misleading product constraint,
or an explanation burden that reveals real causal reasoning.

Do not create a benchmark whose central exploit can be described as "change this
one line/default/flag/operator and the visible test passes." Do not rely on
instructions that forbid the cheap patch. Build benchmark substance: a richer
causal situation and meaningful invariants in the ordinary code and tests.
Explain why those artifacts proxy the target ability in judge_artifact, not in
candidate-visible files. Warnings are not pressure. Negative controls are not
pressure unless they reflect real failure modes a judge can verify from private
metadata and candidate behavior.

Never leak the answer in candidate-facing material. If the benchmark asks an
agent to infer, diagnose, judge, repair, or discover something, the prompt,
inputs, code comments, labels, filenames, visible outputs, fixtures, and setup
must not reveal or strongly hint at the intended answer. A case that gives away
its own answer cannot be promoted no matter how strong the rubric looks.

The JSON boundary is part of the benchmark contract. Everything in
agent_artifact is seen by the evaluated agent at runtime. Everything in
judge_artifact is unseen judge/rubric metadata. Put diagnosis, intended causal
mechanism, expected repair characteristics, scoring rationale, negative-control
explanations, and gaming/shortcut analysis in judge_artifact only. Never copy
judge-facing answers into agent_artifact.prompt, setup, workspace comments,
README text, fixture names, visible test names, or visible outputs.
Candidate-visible code, tests, docs, fixtures, and filenames must look like a
normal project workspace, not an authored benchmark, exercise, postmortem,
answer key, or grader note. The agent-facing prompt may give ordinary task
instructions and a public symptom, but source files and tests must remain
unassuming. Test names, assertion messages, comments, and README prose must not
teach the diagnosis, locate the fault, explain the expected repair, or label the
bug.
Use the private judge fields explicitly: private_root_cause for the hidden diagnosis,
expected_fix_properties for the repair boundary, hidden_failure_modes for checks that
should catch bad fixes, shallow_solution_traps for tempting but invalid repairs, and
candidate_visibility_boundaries for what the evaluated agent must not be shown.

Prefer benchmark designs that create pressure toward excellent outputs, not only
filters against bad outputs. The case should make a strong model reveal taste,
judgment, control, design, or depth that an adequate model would not show.
Avoid converging on familiar safe templates. If the first design is a standard
constraint-following prompt, improve it before returning it by adding meaningful
tradeoff, transformation, preservation, comparison, revision, or other structure
that creates ceiling pressure.

Hard checks may disqualify bad outputs, but table-stakes compliance is not the same as
high ability. A useful benchmark should create evidence about ability, including signals
that can separate adequate from excellent behavior when the domain supports that.

Return only a case you would be willing to defend as a promoted corpus item under
its stated assumptions and limits.
""".strip()


GENERATOR_IMPLEMENTATION_CONTRACT = """

DESIGN IMPLEMENTATION CONTRACT
You are implementing a diagnostic design brief. The design brief decides the target ability, target environment, failure family, diagnostic pressure, shallow paths, and minimum depth. You may choose concrete artifact details, but you may not lower the ambition, simplify the causal structure, replace the failure family, or turn the brief into a smaller benchmark. The design brief's runtime_requirements are binding: do not silently change language, runtime, OS assumptions, dependency policy, package requirements, network posture, or test command shape. If the task needs files, services, tools, packages, or a runtime to execute, declare those requirements in agent_artifact.runtime_requirements and make the environment artifact consistent with them. If implementation choices are needed, choose the strongest faithful artifact that preserves the design's requirements; do not optimize for the smallest possible repository or easiest local patch. Treat agent_artifact as the only material the evaluated agent will see. It must look like an ordinary project workspace plus an ordinary task prompt. Do not make source files, tests, README text, fixture names, assertion messages, or comments explain the benchmark, expose the root cause, label the bug, or point to the repair. Treat judge_artifact as unseen evaluator metadata; put the intended diagnosis, causal explanation, expected repair boundaries, negative-control explanations, and gaming analysis there instead of in candidate-visible materials. Fill private_root_cause, expected_fix_properties, hidden_failure_modes, shallow_solution_traps, and candidate_visibility_boundaries with the hidden information you are tempted to explain in visible code or tests.
""".rstrip()


ADVERSARY_ATTACK_TYPE_TAXONOMY: dict[str, dict[str, str]] = {
    "answer_leakage": {
        "coverage_group": "explicit",
        "definition": "Candidate-facing material reveals or strongly hints at the intended answer, root cause, fix, scoring answer, or hidden expectation.",
    },
    "cheap_pass": {
        "coverage_group": "explicit",
        "definition": "A weak model can get a high score through a shallow or local exploit such as patching a visible line, matching a traceback, changing a literal/operator/default, editing a test, hard-coding an expected value, or following labels and names that reveal the solution.",
    },
    "test_overfitting": {
        "coverage_group": "explicit",
        "definition": "Visible tests or fixtures are narrow enough that fixture-specific hard-coding or test-output overfitting can pass without demonstrating the target ability.",
    },
    "test_overfitting_loophole": {
        "coverage_group": "explicit",
        "definition": "Visible tests or fixtures are narrow enough that fixture-specific hard-coding or test-output overfitting can pass without demonstrating the target ability.",
    },
    "straw_negative_control": {
        "coverage_group": "explicit",
        "definition": "Negative controls are generic, implausible, or too obviously bad rather than concrete plausible shallow fixes tied to the artifact.",
    },
    "proxy_overclaim": {
        "coverage_group": "indirect",
        "definition": "The proxy claim overstates what the concrete artifact can actually prove about the target ability.",
    },
    "fake_difficulty": {
        "coverage_group": "indirect",
        "definition": "The benchmark looks complex but the real repair or success path is decorative, local, or toy-shaped.",
    },
    "non_discriminating_regression": {
        "coverage_group": "indirect",
        "definition": "Tests or regressions do not distinguish the intended fix from plausible shallow or semantically wrong fixes.",
    },
    "scoring_ambiguity": {
        "coverage_group": "uncovered",
        "definition": "Candidate-facing requirements and judge-facing scoring criteria disagree, omit necessary invariants, or require optional/manual/private checks that are not enforced by the concrete artifact.",
    },
    "other": {
        "coverage_group": "uncovered",
        "definition": "An attack that does not fit the named taxonomy.",
    },
}


REJECT_SIGNAL_CODES = {
    "answer_leak_explicit_bug_label",
    "answer_leak_root_cause_disclosure",
    "answer_leak_fix_instruction",
    "answer_leak_fault_location",
    "answer_leak_fault_behavior_description",
    "answer_leak_intended_behavior",
    "answer_leak_scaffold_label",
    "weak_proxy_validity",
    "unreliable_score",
    "weak_diagnostic_pressure",
    "shortcut_leakage",
    "vague_scoring_contract",
    "fake_difficulty",
    "irrelevant_environment",
    "ambiguous_success_criteria",
    "overbroad_proxy_claim",
    "missing_known_limits",
    "missing_negative_control",
    "missing_oracle",
    "schema_violation",
    "near_duplicate",
}


GENERATOR_RETRY_CODE_MAP = {
    "missing_private_oracle": "weak_judge_confidence",
    "missing_oracle": "weak_judge_confidence",
    "private_oracle_integrity": "weak_judge_confidence",
}


GENERATOR_RETRY_GUIDANCE = {
    "workspace_tests_do_not_reproduce_failure": (
        "The starter workspace pytest command passed. Regenerate complete files so the unmodified starter code has at least one deterministic failing pytest assertion that demonstrates the target failure. "
        "Do not make every test fail; include enough passing tests to show the harness is otherwise healthy."
    ),
    "workspace_test_command_failed": (
        "The workspace test command did not cleanly collect and run to pytest assertions. Fix syntax, imports, pytest config, and file completeness so pytest executes normally."
    ),
    "answer_leak_explicit_bug_label": (
        "Candidate-facing material labels the workspace as buggy or intentionally defective. Remove exercise-author bug labels such as BUG comments and intentionally-buggy prose."
    ),
    "answer_leak_root_cause_disclosure": (
        "Candidate-facing material names the root cause or causal region. Move root-cause analysis into private judge fields."
    ),
    "answer_leak_fix_instruction": (
        "Candidate-facing material tells the evaluated agent what fix to make. Move expected fix properties into private judge fields."
    ),
    "answer_leak_fault_location": (
        "Candidate-facing material points too directly at the faulty line, function, expression, file, or fixture. Make localization require diagnosis."
    ),
    "answer_leak_fault_behavior_description": (
        "Candidate-facing material describes how the current implementation is wrong. Keep normal domain semantics visible, but move fault analysis into private judge fields."
    ),
    "answer_leak_intended_behavior": (
        "Candidate-facing material states the intended invariant or behavior in answer-key form. Express only user-visible symptoms publicly and keep hidden invariants private."
    ),
    "answer_leak_scaffold_label": (
        "Candidate-facing material frames the task as starter code or an exercise scaffold. Make the workspace read like a realistic project artifact."
    ),
    "weak_diagnostic_pressure": (
        "The artifact did not create enough diagnostic pressure. Strengthen the actual workspace behavior with interacting components, misleading symptoms, and tests that distinguish shallow patches from causal fixes."
    ),
    "weak_proxy_validity": (
        "The artifact was not a strong proxy for the claimed ability. Make the visible workspace itself require the target ability; do not rely on judge-facing prose to create the difficulty."
    ),
}


DESIGN_RETRY_GUIDANCE = {
    "missing_runtime_requirements": (
        "Add runtime_requirements to every design. Executable filesystem tasks need kind=filesystem_task, execution, language, dependencies, commands.test, and network posture."
    ),
    "unsupported_runtime_requirements": (
        "For executable filesystem tasks, set runtime_requirements.execution.mode to exactly 'task_image' or exactly 'container' and include execution.base_image. Do not return combined strings like 'task_image/container'."
    ),
}
