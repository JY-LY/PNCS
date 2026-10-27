import torch
import torch.nn as nn
import torch.nn.functional as F


class PNCS(nn.Module):
    def __init__(self, args, base_encoder,cla_embedding):
        super().__init__()

        pretrained = False
        
        self.proto_weight = args.proto_m

        self.encoder = base_encoder(name=args.arch, head='mlp', feat_dim=768, num_class=args.num_class, pretrained=pretrained)

        self.register_buffer("prototypes", cla_embedding)
    
    
    def set_prototype_update_weight(self, epoch: object, args: object) -> object:
        start = args.pro_weight_range[0]
        end = args.pro_weight_range[1]
        self.proto_weight = 1. * epoch / args.epochs * (end - start) + start

    @torch.no_grad()
    def _momentum_update_prototypes(self, features, labels):
        features = features.detach()
        labels = labels.detach().long()

        if labels.numel() == 0:
            return

        num_classes = self.prototypes.size(0)

        proto_weight = torch.full((), self.proto_weight, dtype=features.dtype, device=features.device)

        one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(features.dtype)
        occurrence_index = one_hot_labels.cumsum(dim=0)[torch.arange(labels.numel(), device=labels.device), labels]
        label_counts = one_hot_labels.sum(dim=0)
        active_labels = label_counts > 0

        update_exponents = label_counts[labels] - occurrence_index
        update_weights = torch.pow(proto_weight, update_exponents).unsqueeze(1)

        weighted_features = features * update_weights
        feature_updates = torch.zeros_like(self.prototypes)
        feature_updates.index_add_(0, labels, weighted_features)

        prototype_weights = torch.pow(proto_weight, label_counts).unsqueeze(1)
        updated_prototypes = (
            prototype_weights * self.prototypes
            + (1 - proto_weight) * feature_updates
        )
        self.prototypes.copy_(torch.where(
            active_labels.unsqueeze(1), updated_prototypes, self.prototypes
        ))


    def forward(self, img_q, img_k=None, img_q_mix=None, img_k_mix=None, partial_Y=None, Y_true=None, args=None, eval_only=False, run_epoch=None):
        
        output_q, q = self.encoder(img_q)
        
        if eval_only:
            prototypes_eval = self.prototypes.detach()
            logits_prot_test = torch.mm(q, prototypes_eval.t())
            return output_q, logits_prot_test
        
        output_k, k = self.encoder(img_k)

        output_q_mix, q_mix = self.encoder(img_q_mix)
        output_k_mix, k_mix = self.encoder(img_k_mix)

        with torch.no_grad():
            predicetd_scores_q = torch.softmax(output_q, dim=1) * partial_Y
            predicetd_scores_q_norm = predicetd_scores_q / predicetd_scores_q.sum(dim=1, keepdim=True)

            predicetd_scores_k = torch.softmax(output_k, dim=1) * partial_Y
            predicetd_scores_k_norm = predicetd_scores_k / predicetd_scores_k.sum(dim=1, keepdim=True)

            max_scores_q, pseudo_labels_q = torch.max(predicetd_scores_q_norm, dim=1)
            max_scores_k, pseudo_labels_k = torch.max(predicetd_scores_k_norm, dim=1)
        
        prototypes = self.prototypes.clone().detach()

        logits_prot_q = torch.mm(q, prototypes.t())
        logits_prot_k = torch.mm(k, prototypes.t())


        logits_prot_q_mix = torch.mm(q_mix, prototypes.t())
        logits_prot_k_mix = torch.mm(k_mix, prototypes.t())

        gathered_features = torch.cat((concat_all_gather(q), concat_all_gather(k)), dim=0)
        gathered_labels = torch.cat((concat_all_gather(pseudo_labels_q), concat_all_gather(pseudo_labels_k)), dim=0)
        self._momentum_update_prototypes(gathered_features, gathered_labels)

        self.prototypes.copy_(F.normalize(self.prototypes, p=2, dim=1))
        
        return output_q, output_k, output_q_mix, output_k_mix, logits_prot_q, logits_prot_k, logits_prot_q_mix, logits_prot_k_mix



# From https://github.com/hbzju/PiCO/blob/main/model.py
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor.detach()

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensor.detach()

    tensors_gather = [torch.empty_like(tensor)
        for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather, dim=0)
    
    return output


@torch.no_grad()
def get_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-7) # avoid NaN
    return -(probs*log_probs).sum(1)
