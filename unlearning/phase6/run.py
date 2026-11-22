from phase2.evaluate import validate_state
from phase1.analyze import transform_batch
import task_template as official


def state_with_head(reference, head):
    return {**reference, "head.weight": head[:, :-1].contiguous(), "head.bias": head[:, -1].contiguous()}
