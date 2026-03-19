# AGENTS.md

## Purpose

This repository is a reinforcement-learning-based agent project.  
The primary goal of the agent is to make safe, minimal, and correct changes within this repository only.

This file defines strict execution, filesystem, and reporting rules for AI coding agents.

---

## Highest-Priority Rules

1. **Do not modify, create, move, or delete anything outside this repository workspace.**
2. **Never touch other users' directories, shared lab directories, or unrelated system paths.**
3. **Before finishing a task, always report how the user can determine whether the task has completed successfully.**
4. **Do not treat a submitted long-running command as "done" just because it was launched.**
5. **Prefer safe, minimal edits over broad refactors.**
For model servers (vLLM, inference endpoints), always provide:
- start command
- stop command
- health check command
If any instruction conflicts with these rules, these rules take priority.

---

## Workspace Boundary Rules

Assume that this machine is a shared lab or multi-user server.

### Allowed
- Read and write files **inside this repository only**
- Create project-local files inside this repository
- Modify source code, configs, scripts, and docs inside this repository
- Use project-local virtual environment / conda environment / caches if already configured for this project
- Read from user-owned development resources such as conda, huggingface cache, pip cache, or similar dependency locations **only when necessary to run this project**

### Forbidden
- Do not create files outside the repository
- Do not delete files outside the repository
- Do not move or rename files outside the repository
- Do not create or delete directories outside the repository
- Do not modify other users' home directories
- Do not modify shared lab folders unless the user explicitly names a project-owned path inside them
- Do not run destructive cleanup commands on broad paths

### Never run commands like
- `rm -rf /`
- `rm -rf ~`
- `rm -rf ../`
- `rm -rf /tmp/*`
- `find .. -delete`
- `mv * /somewhere`
- any recursive delete or move that is not clearly confined to this repository

### Path Safety Policy
Before any command that writes, deletes, moves, or renames files:
1. Confirm the target path is inside the current repository
2. Prefer explicit paths over wildcards
3. Prefer file-by-file edits over directory-wide operations
4. Refuse risky commands if the affected scope is unclear

When cleaning artifacts, only remove clearly project-local outputs such as:
- `./outputs/`
- `./runs/`
- `./logs/`
- `./checkpoints_tmp/`

and only when those directories are part of this repository.

---

## Shared Machine Safety Rules

This machine may contain:
- other users' home directories
- shared datasets
- shared experiment outputs
- shared model caches
- shared workspaces

Therefore:

- Never assume a parent directory is safe
- Never assume sibling directories belong to this project
- Never create "helper" folders outside the repo
- Never delete old experiment folders unless they are definitely inside this repo
- Never reorganize files outside the repo for convenience

If a task seems to require writing outside the repo, stop and ask the user.

---

## Command Execution Policy

### General
- Prefer short, inspectable commands
- Prefer project-local scripts over ad hoc shell pipelines
- Avoid destructive shell one-liners
- Avoid backgrounding a job unless necessary
- Do not claim success only because a command was launched

### Long-Running Jobs
Training, evaluation, serving, and rollout jobs may take a long time.

When launching a long-running job, the agent must do all of the following:

1. State clearly that the job has been **started**, not completed
2. Provide the exact command used
3. Provide the expected output location(s)
4. Provide the exact log file location(s)
5. Provide the exact completion criteria
6. Provide the exact failure signals the user should check
7. If possible, provide a safe command the user can run to inspect progress

### Required End-of-Task Handoff for Long Jobs
If the task involves a long-running process, the final response must include a section named:

`How to tell whether the job is finished`

That section must include:
- process completion signal
- expected output files
- expected final log lines or metrics
- how to distinguish success from failure
- where checkpoints / result artifacts should appear

Do not end with vague statements like:
- "the job is running"
- "it should finish later"
- "check it after some time"

Be concrete.

---

## Completion Reporting Standard

Before ending the conversation, always report task status using one of these labels:

- **Completed**: the requested work is fully done and verified
- **Started, still running**: a long-running job has been launched but has not finished
- **Blocked**: progress stopped due to a real constraint
- **Partial**: some requested work is done, but not all

If status is **Started, still running**, include all of:
- launch command
- PID if available
- log file path
- output path
- completion criteria
- failure criteria
- next safe inspection command

---

## Example Reporting Template for Long Jobs

Use a structure like this in the final response:

### Status
Started, still running

### Command used
`python train.py --config configs/train.yaml`

### Log file
`./logs/train_run_001.log`

### Expected outputs
- `./checkpoints/run_001/last.ckpt`
- `./checkpoints/run_001/best.ckpt`
- `./results/run_001/metrics.json`

### How to tell whether the job is finished
- The process no longer appears in `ps`
- The log ends with a clear completion message such as training finished / evaluation complete
- The final checkpoint file exists
- The metrics file has been written successfully

### Signs of failure
- traceback in the log
- OOM / CUDA out of memory
- NaN loss
- missing checkpoint or metrics files

### Safe progress check
`tail -n 50 ./logs/train_run_001.log`

---

## Code Change Policy

- Prefer modifying existing files over creating new abstractions
- Keep changes minimal and local
- Do not perform broad refactors unless explicitly requested
- Do not rename public interfaces unless necessary
- Do not introduce new frameworks or major dependencies without explicit user approval

For RL / agent code specifically:
- preserve existing training pipeline unless the task requires changing it
- preserve config structure unless the task requires changing it
- prefer explicit config changes over hidden behavior changes
- keep experiment logic reproducible

---

## File Creation Rules

Allowed inside the repository:
- source files
- config files
- logs
- checkpoints
- results
- temporary project-local scripts

Not allowed outside the repository:
- temporary helper scripts
- scratch files
- shell scripts
- experiment outputs
- copied datasets
- copied checkpoints

---

## Deletion Rules

Deletion must be conservative.

Allowed:
- remove temporary files created for the current task inside this repository
- remove clearly disposable project-local artifacts when necessary

Not allowed:
- deleting unknown directories
- deleting parent/sibling directories
- deleting shared caches unless the user explicitly asks
- deleting any path whose ownership or scope is unclear

When uncertain, do not delete.

---

## Environment Rules

- Avoid sudo unless the user explicitly requests system-level setup
- Prefer user-space tooling
- Prefer project-local environments
- Do not change global system configuration
- Do not install system packages unless explicitly requested
- Do not assume full machine ownership

Conda, huggingface cache, pip cache, CUDA tooling, and similar dependencies may be read when needed, but should not be reorganized, deleted, or broadly cleaned.

---

## Logging and Outputs

For any training / evaluation / serving task:
- prefer explicit log redirection
- prefer deterministic output paths
- state where artifacts will appear
- state what file proves success

Good:
- `./logs/...`
- `./checkpoints/...`
- `./results/...`

Avoid scattering outputs across random directories.

---

## When Unsure

Ask before:
- writing outside the repository
- deleting directories
- changing shared paths
- using sudo
- changing global environment settings
- running broad cleanup commands
- launching very long jobs without a clear log/output path

When uncertain, choose safety over convenience.
## Execution Delegation Policy

Some operations are better executed by the user directly, especially on shared machines.

The agent should NOT insist on executing every command itself.

If a command involves:

- starting long-running services (e.g., vLLM, training loops, model servers)
- occupying GPUs
- binding network ports
- requiring sudo or elevated privileges
- background processes that may run for hours
- interactive or environment-sensitive commands

then the agent may **delegate execution to the user**.

In that case the agent must:

1. Provide the exact command to run.
2. Ensure the command is copy-paste ready.
3. Explain what the command does.
4. Explain how to verify that the process started correctly.
5. Explain how to stop the process if needed.

The agent should clearly label this as:

`User should run the following command`

The agent must not treat delegated execution as task completion.

Instead it should report the task status as:

`Waiting for user execution`