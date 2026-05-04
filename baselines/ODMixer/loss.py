from basicts.metrics import masked_mae


def odmixer_loss(prediction, target, prev_prediction=None, prev_target=None, null_val=float("nan")):
    loss = masked_mae(prediction, target, null_val)
    if prev_prediction is not None and prev_target is not None:
        loss = loss + masked_mae(prev_prediction, prev_target, null_val)
    return loss
