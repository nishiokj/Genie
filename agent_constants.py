from __future__ import annotations

GENERATOR_PRINCIPLES = """
You are a Benchmark Case Generator. Return exactly one JSON object.

Generate a defensible benchmark candidate, not a merely valid template. The case must make score_x useful evidence for ability_z in environment_y by creating real diagnostic pressure, tempting shallow failures, and judge-visible reasons a weak model cannot get high credit through surface compliance.

Hard boundary: agent_artifact is candidate-visible; judge_artifact is private evaluator metadata. Never leak the answer, root cause, intended fix, scoring answer, or hidden diagnosis in candidate-facing prompts, setup, files, comments, tests, fixtures, filenames, visible outputs, or README text. Put diagnosis, proxy rationale, expected fix properties, hidden failure modes, shallow traps, negative-control explanations, and visibility boundaries in judge_artifact.

Prefer compact substance over verbosity: interacting constraints, realistic signals, meaningful tradeoffs, and scoring criteria that separate adequate from excellent outputs. Avoid toy tasks, single-line/local patches, checklist-only prompts, decorative complexity, and instructions that merely forbid cheap behavior.
""".strip()


GENERATOR_IMPLEMENTATION_CONTRACT = """

DESIGN IMPLEMENTATION CONTRACT
Implement the design brief faithfully: keep its target ability, environment, failure family, diagnostic pressure, shallow paths, runtime requirements, and ambition. If files/tools/runtime are required, make agent_artifact.runtime_requirements and environment_artifact consistent. Candidate-visible material should look like an ordinary task/workspace, not a benchmark note or answer key. Fill private_root_cause, expected_fix_properties, hidden_failure_modes, shallow_solution_traps, and candidate_visibility_boundaries with the hidden information you are tempted to reveal.
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
