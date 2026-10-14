# Running the SLM project on the GPU node

Everything in this project runs on the **compute** GPU node (8 × RTX A6000, 48 GB), never on master.
You don't need `qsub`, `ssh compute`, or `CUDA_VISIBLE_DEVICES` by hand. The project handles them.

---

## 1. Run a notebook in VS Code (everyday use)

1. Connect VS Code to **master** with Remote-SSH and open the folder `/home/urvashib/projects/SLM`.
2. Open any notebook in this folder.
3. Kernel at the top right → **SLM .venv (GPU compute node)**.
   (The project notebook selects it automatically. For a new notebook, choose it once:
   **Select Kernel → Jupyter Kernel... → SLM .venv (GPU compute node)**.)
4. Click **Run All**, or run cells one by one.

The first cell prints:

```
Node: compute
Python: /home/urvashib/projects/SLM/.venv/bin/python
CUDA: True
GPU count: 1
GPU 0: NVIDIA RTX A6000 (47.4 GB)
```

What happens behind the scenes:

| You do | It does |
|---|---|
| Select the kernel or start the notebook | Submits a PBS job `slm-kernel` to `GPU_only` and starts Python there on the least-busy GPU (takes a few seconds) |
| Interrupt (■) | Stops the running cell |
| Restart kernel | Ends the old job and starts a fresh one |
| Close the notebook, close VS Code, or lose your connection | The PBS job ends and the GPU is freed |

`GPU count: 1` is expected. Each notebook gets one GPU of its own, picked automatically.

> Only 2 GPU jobs per user are allowed. Each open notebook kernel counts as 1 job.
> If a kernel won't start, close other notebooks or run `qstat -u $USER` and `qdel` old jobs.

## 2. Run a `.py` file on the GPU

- **From VS Code:** open the file and press `Ctrl+Shift+B` ("Run current file on GPU node").
- **From the terminal (on master):**
  ```bash
  cd /home/urvashib/projects/SLM
  scripts/gpurun python my_script.py
  ```
- **Line by line:** select code in a `.py` file and press `Shift+Enter`. It runs in the
  Interactive Window on the GPU kernel.

> The ▶ "Run Python File" button runs on **master** (no GPU). Use `Ctrl+Shift+B` instead.

## 3. Long training run (keeps going after you log out)

```bash
cd /home/urvashib/projects/SLM
qsub run_notebook.pbs        # runs the whole notebook as a batch job
qstat -u $USER               # slm-train: R = running, gone = finished
tail slm-train.log           # ends with "Finished: ..."
```

Results: `notebooks/slm-output.ipynb` (all outputs) and `notebooks/best_model_params.pt` (weights).

---

## Checking and cleaning up jobs

```bash
qstat -u $USER                                    # your jobs: slm-kernel / slm-run / slm-train
qstat -f <JobID> | grep -E "Job_Name|job_state"   # details of one job
qdel <JobID>                                      # stop a job
ssh compute nvidia-smi                            # GPU usage
```

## How it is set up (for reference)

| File | Purpose |
|---|---|
| `~/.local/share/jupyter/kernels/slm-venv/kernel.json` | The "SLM .venv (GPU compute node)" kernel; it calls `scripts/gpu_kernel` |
| `scripts/gpu_kernel` | Starts the kernel in a PBS job on compute and tunnels it back to VS Code |
| `scripts/gpurun` | Runs any command on compute: `scripts/gpurun <command>` |
| `.vscode/settings.json` | Hides the plain `.venv` (master, no GPU) from the kernel list |
| `.vscode/tasks.json` | `Ctrl+Shift+B` = run the current file on compute |
| `run_notebook.pbs` | Batch job for a full training run |
