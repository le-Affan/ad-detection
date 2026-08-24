"""Measure the largest batch size that actually fits on this GPU.

Guessing a batch size for a 33.2M-parameter 3D ResNet-18 on 96^3 volumes is
how a run dies at epoch 0 with CUDA OOM after the dataset has already mounted.
This runs a real training step -- forward, backward, AND optimizer.step() --
for each candidate and reads peak memory back from the allocator.

THE OPTIMIZER STEP IS NOT OPTIONAL IN THE MEASUREMENT. AdamW lazily allocates
exp_avg and exp_avg_sq on its first step, 2x the parameter count, ~265 MB here.
Probing forward+backward only would under-report peak memory by that much and
recommend a batch size that OOMs on the first optimizer step of the real run.

WHAT IS REPORTED. `reserved` (not `allocated`) is the number that matters:
the caching allocator holds reserved blocks, and a later allocation fails
against reserved, not against the sum of live tensors. The recommendation
keeps peak reserved under --safety of total VRAM (default 0.85), leaving room
for fragmentation and for the eval pass, which builds no grads but does hold
activations.

Usage on Kaggle, before the real run:

    python src/mri/m3_probe_batch.py                 # 4, 8, 16, 24, 32
    python src/mri/m3_probe_batch.py --sizes 8,16    # narrower sweep
    python src/mri/m3_probe_batch.py --no-amp        # if training without AMP

It allocates synthetic volumes rather than reading the dataset, so it needs no
data mounted and takes a few seconds.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DualHeadNet, dual_head_loss  # noqa: E402

SHAPE = (96, 96, 96)


def probe_one(bs, device, amp, shape=SHAPE, steps=2):
    """Run `steps` full training steps at batch size `bs`; return peak memory.

    Returns (ok, reserved_bytes, allocated_bytes, sec_per_step, err).
    Two steps, because step 1 allocates the AdamW state and step 2 is what a
    steady-state iteration actually costs.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = opt = None
    try:
        model = DualHeadNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        x = torch.randn(bs, 1, *shape, device=device)
        tgt = {"demo": torch.rand(bs, device=device),
               "alz": torch.randint(0, 2, (bs,), device=device).float()}

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(steps):
            if i == steps - 1:
                torch.cuda.synchronize()
                start.record()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
                loss, _ = dual_head_loss(out, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000.0

        res = torch.cuda.max_memory_reserved()
        alloc = torch.cuda.max_memory_allocated()
        return True, res, alloc, sec, None
    except torch.cuda.OutOfMemoryError as e:
        return False, 0, 0, 0.0, "OOM"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, 0, 0, 0.0, "OOM"
        raise
    finally:
        del model, opt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default="4,8,16,24,32")
    ap.add_argument("--safety", type=float, default=0.85,
                    help="max fraction of total VRAM the recommendation may use")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--train-subjects", type=int, default=567,
                    help="only used to estimate epoch time")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- this probe measures GPU memory and cannot")
        print("run on CPU. Run it on the Kaggle GPU session before training.")
        return 1

    device = torch.device("cuda")
    amp = not args.no_amp
    total = torch.cuda.get_device_properties(0).total_memory
    print("gpu    : {}".format(torch.cuda.get_device_name(0)))
    print("vram   : {:.2f} GiB".format(total / 2**30))
    print("amp    : {}".format(amp))
    print("budget : {:.2f} GiB ({:.0%} of total)\n".format(
        total * args.safety / 2**30, args.safety))

    print("{:>6} {:>12} {:>12} {:>9} {:>10} {:>9}".format(
        "batch", "reserved", "allocated", "% total", "sec/step", "verdict"))
    best = None
    for bs in [int(s) for s in args.sizes.split(",") if s.strip()]:
        ok, res, alloc, sec, err = probe_one(bs, device, amp)
        if not ok:
            print("{:>6} {:>12} {:>12} {:>9} {:>10} {:>9}".format(
                bs, "-", "-", "-", "-", err))
            break  # larger sizes cannot fit either
        frac = res / total
        fits = frac <= args.safety
        print("{:>6} {:>11.2f}G {:>11.2f}G {:>8.1%} {:>10.3f} {:>9}".format(
            bs, res / 2**30, alloc / 2**30, frac, sec, "ok" if fits else "over budget"))
        if fits:
            best = (bs, sec)

    print()
    if best is None:
        print("RECOMMENDATION: none of the probed sizes fit under the safety margin.")
        print("Try smaller sizes, e.g. --sizes 1,2,4")
        return 1
    bs, sec = best
    steps = (args.train_subjects + bs - 1) // bs
    print("RECOMMENDATION: --batch-size {}".format(bs))
    print("  {:.3f} s/step x {} steps ~= {:.1f} s/epoch of training".format(
        sec, steps, sec * steps))
    print("  (eval adds roughly a third of that; no-augmentation loading is the rest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
