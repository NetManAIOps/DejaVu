import torch as th
import torch.jit
import torch.nn.functional as func


@torch.jit.script
def focal_loss(preds, labels, gamma: float = 2.):
    preds = preds.view(-1, preds.size(-1))  # [-1, num_classes]
    labels = labels.view(-1, 1)  # [-1, ]
    preds_logsoft = func.log_softmax(preds, dim=-1)  # log_softmax
    preds_softmax = th.exp(preds_logsoft)  # softmax
    preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
    preds_logsoft = preds_logsoft.gather(1, labels)
    weights = th.pow((1 - preds_softmax), gamma)
    loss = - weights * preds_logsoft  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

    loss = th.sum(loss) / th.sum(weights)
    return loss


# @torch.jit.script
def binary_classification_loss(
        prob: th.Tensor, label: th.Tensor, gamma: float = 0.,
        normal_fault_weight: float = 1e-1,
        target_node_weight: float = 0.5,
) -> th.Tensor:
    assert prob.size() == label.size()
    device = prob.device
    target_id = th.argmax(label, dim=-1)
    prob_softmax = th.sigmoid(prob)
    weights = th.pow(
        th.abs(label - prob_softmax), gamma
    ) * (
                      th.max(label, dim=-1, keepdim=True).values + th.full_like(label, fill_value=normal_fault_weight)
              )
    if len(prob.size()) == 1:
        weights[target_id] *= prob.size()[-1] * target_node_weight
    else:
        weights[th.arange(len(target_id), device=device), target_id] *= prob.size()[-1] * target_node_weight
    loss = func.binary_cross_entropy_with_logits(prob, label.float(), reduction='none')
    return th.sum(loss * weights) / th.prod(th.tensor(weights.size()))


@torch.jit.script
def multi_classification_loss(prob: th.Tensor, label: th.Tensor, gamma: float = 2) -> th.Tensor:
    assert prob.size() == label.size()
    target_id = th.argmax(label, dim=-1)
    assert len(prob.size()) == len(target_id.size()) + 1
    if len(prob.size()) == 1:
        return focal_loss(prob.view(1, -1), target_id.view(1), gamma=gamma)
    else:
        return focal_loss(prob, target_id, gamma=gamma)
