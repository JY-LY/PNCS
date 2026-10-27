import torch
import torch.nn as nn


class PNCSLoss(nn.Module):
    def __init__(self, predicted_score_cls, pseudo_label_weight = 0.99):
        super().__init__()
        self.predicted_score_cls1 = predicted_score_cls
        self.predicted_score_cls2 = predicted_score_cls

        self.init_predicted_score_cls = predicted_score_cls.detach()

        self.pseudo_label_weight = pseudo_label_weight
    

    def set_alpha(self, epoch, args):
        self.alpha = min((epoch / 10) * args.alpha_weight, args.alpha_weight)

    def set_pseudo_label_weight(self, epoch, args):
        start = args.pseudo_label_weight_range[0]
        end = args.pseudo_label_weight_range[1]
        self.pseudo_label_weight = 1. * epoch / args.epochs * (end - start) + start

    def get_pl_alpha(self, epoch):

        if epoch < 50:
            return 0
        else:
            return 1

    def update_weight_byclsout1(self, cls_predicted_score, batch_index, batch_partial_Y, args):
        with torch.no_grad():
            y_pred_raw_probas = torch.softmax(cls_predicted_score, dim = 1)
            
            revisedY_raw = batch_partial_Y * y_pred_raw_probas
            revisedY_raw = revisedY_raw / revisedY_raw.sum(dim=1, keepdim=True)
            
            cls_pseudo_label = revisedY_raw.detach()
            
            self.predicted_score_cls1[batch_index, :] = self.pseudo_label_weight * self.predicted_score_cls1[batch_index, :] + \
                                                    (1 - self.pseudo_label_weight) * cls_pseudo_label
            

    def forward(self, cls_out1, cls_out_mix1, cls_out_mix2, logits_prot1, logits_prot_1_mix, logits_prot_2_mix, idx_rp, Lambda, index, args, sim_criterion, epoch):

        y_pred_1_probas = torch.softmax(cls_out1, dim = 1)
        
        prot_pred_1_mix_probas_log = torch.log_softmax(logits_prot_1_mix / args.tau_proto, dim = 1)
        prot_pred_2_mix_probas_log = torch.log_softmax(logits_prot_2_mix / args.tau_proto, dim = 1)

        pred_mix_1_logits = torch.log_softmax(cls_out_mix1 / args.tau_proto, dim=1)
        pred_mix_2_logits = torch.log_softmax(cls_out_mix2 / args.tau_proto, dim=1)


        soft_positive_label_target1 = self.predicted_score_cls1[index, :].clone().detach()
        soft_positive_label_target1_rp = soft_positive_label_target1[idx_rp]

        pl_alpla = self.get_pl_alpha(epoch)

        cls_loss_all_1 = soft_positive_label_target1 * torch.log(y_pred_1_probas)
        cls_loss_1 = - ((cls_loss_all_1).sum(dim=1)).mean()

        mask_positive = soft_positive_label_target1 > 0
        negative_soft_positive = torch.where(mask_positive, 1 - soft_positive_label_target1,soft_positive_label_target1)
        negative_soft_loss = -torch.mean(torch.sum(torch.log(1.0000001 - y_pred_1_probas) * negative_soft_positive, dim=1))

        final_loss = cls_loss_1 + pl_alpla * negative_soft_loss

        negative_mask = soft_positive_label_target1 == 0

        weighted_positive_similarity = (logits_prot1 * soft_positive_label_target1).sum(dim=1)
        mean_pos_sim = weighted_positive_similarity.mean().detach()

        dynamic_margin = args.mu + 0.05 * mean_pos_sim

        negative_similarity = logits_prot1 * negative_mask
        contrastive_loss = torch.clamp((dynamic_margin - weighted_positive_similarity.unsqueeze(dim=1) + negative_similarity) * negative_mask,min=0).sum(dim=1).mean()

        sim_loss_2_1 = Lambda * sim_criterion(prot_pred_1_mix_probas_log, soft_positive_label_target1) + \
                    (1 - Lambda) * sim_criterion(prot_pred_1_mix_probas_log, soft_positive_label_target1_rp)

        sim_loss_2_2 = Lambda * sim_criterion(prot_pred_2_mix_probas_log, soft_positive_label_target1) + \
                    (1 - Lambda) * sim_criterion(prot_pred_2_mix_probas_log, soft_positive_label_target1_rp)

        sim_loss_3 = Lambda * sim_criterion(pred_mix_1_logits, soft_positive_label_target1) + \
                    (1 - Lambda) * sim_criterion(pred_mix_1_logits, soft_positive_label_target1_rp)

        sim_loss_4 = Lambda * sim_criterion(pred_mix_2_logits, soft_positive_label_target1) + \
                    (1 - Lambda) * sim_criterion(pred_mix_2_logits, soft_positive_label_target1_rp)


        sim_loss_2 = sim_loss_2_1 + sim_loss_2_2 + sim_loss_3 + sim_loss_4
        

        return final_loss, contrastive_loss, sim_loss_2, self.alpha


