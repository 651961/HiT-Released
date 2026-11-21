"""Calibrate the next update's two HiT thresholds from this update's global batch."""
import torch
import torch.distributed as dist


class MergeHistograms:
    def __init__(self, device, bins=4096):
        # Normalized affinities are in [-1, 1], allowing slight floating-point overshoot.
        self.edges = torch.linspace(-1.00001, 1.00001, bins + 1, dtype=torch.float64, device=device)
        # Last two entries hold the actual input population and retained group count.
        self.totals = torch.zeros((2, bins + 2), dtype=torch.long, device=device)

    @torch.no_grad()
    def add(self, aux):
        first, second = aux['merge_scores']
        counts = aux['hard_counts']
        columns = torch.arange(second.shape[1], device=second.device)
        valid = columns[None, :] < counts[:, None] - 1
        for row, scores, population, retained in (
            (self.totals[0], first.flatten(), counts.numel() * aux['token_count'], counts.sum()),
            (self.totals[1], second[valid], counts.sum(), aux['hard_counts2'].sum()),
        ):
            indices = torch.bucketize(scores.contiguous(), self.edges[1:-1], right=True)
            row[:-2].scatter_add_(0, indices, torch.ones_like(indices))
            row[-2] += population
            row[-1] += retained

    @torch.no_grad()
    def apply(self, net):
        if dist.is_initialized():
            dist.all_reduce(self.totals)
        metrics = {}
        for stage, (router, row, ratio) in enumerate(
                zip((net.router, net.router2), self.totals, net.config.ratios), start=1):
            hist, population, retained = row[:-2], row[-2], row[-1]
            # For threshold edges[i], every score in bins i and above is merged.
            merges = hist.sum() - torch.cat((hist.new_zeros(1), hist.cumsum(0)))
            error = (population - merges - population.double() / ratio).abs()
            # Only consider edges bordering an occupied bin, including both sides of ties.
            occupied = hist > 0
            empty = occupied.new_zeros(1)
            candidates = torch.cat((occupied, empty)) | torch.cat((empty, occupied))
            best = error.masked_fill(~candidates, torch.inf).argmin()
            metrics[f'threshold{stage}_used'] = router.threshold.detach().clone()
            router.threshold.copy_(torch.where(occupied.any(), self.edges[best], router.threshold))
            metrics[f'threshold{stage}_next'] = router.threshold.detach().clone()
            metrics[f'retained_fraction{stage}'] = retained / population
        return metrics
