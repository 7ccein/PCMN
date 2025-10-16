from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F
from .inclusion import InclusionLoss

class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        return mask.int()

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights = self.compute_pool_weights(lengths, features)
        features = features[:, :int(lengths.max()), :]
        pooled_features = (features * pool_weights).sum(1)
        return pooled_features, pool_weights
    
class UncertainLoss(nn.Module):
    def __init__(
            self,
            init_shift=5,
            init_negative_scale=5,
            vib_beta=0,
            smoothness_alpha=0,
            prob_distance='csd',
            **kwargs):
        super().__init__()

        shift = init_shift * torch.ones(1)
        negative_scale = init_negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.vib_beta = vib_beta
        self.smoothness_alpha = smoothness_alpha

        self.bceloss = nn.BCEWithLogitsLoss()

    def _recompute_matched(self, matched, logits, smoothness=0):
        """ Recompute the `matched` matrix if the smoothness value is given.
        """
        if not smoothness:
            return matched, None
        else:
            logits = logits.view(matched.size())
            # XXX Warning: all negative pairs will return weird results
            gt_labels, gt_indices = torch.max(matched, dim=1)
            gt_vals = logits[:, gt_indices].diag()
            pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1))
            new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices))
            _matched = matched.clone()
            _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices]

            return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)

    def _compute_prob_matching_loss(self, logits, matched, smoothness=0):
        matched, n_pseudo_gts = self._recompute_matched(matched, logits, smoothness)
        loss = self.bceloss(logits, matched)

        return {
            'loss': loss,
            'n_pseudo_gts': n_pseudo_gts,
        }

    def _compute_closed_form_loss(self, input1, input2, matched, smoothness=0):
        """ Closed-form probabilistic matching loss -- See Eq (1) and (2) in the paper.
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = ((torch.exp(input1['std']).unsqueeze(1) + torch.exp(input2['std']).unsqueeze(0))).sum(-1)
        logits = mu_pdist + sigma_pdist
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(logits, matched, smoothness=smoothness)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict


    def forward(self, img_emb, cap_emb, matched=None):
        loss_fn = self._compute_closed_form_loss
        vib_loss = 0

        loss = loss_fn(img_emb, cap_emb, matched=matched)
        loss = 2 * loss['loss'] + self.vib_beta * vib_loss

        loss_dict = {
            'loss/loss': loss,
            'criterion/shift': self.shift,
            'criterion/negative_scale': self.negative_scale,
        }

        if self.smoothness_alpha:
            smooth_i2t_loss = loss_fn(img_emb, cap_emb, matched=matched, smoothness=self.smoothness_alpha)
            smooth_t2i_loss = loss_fn(cap_emb, img_emb, matched=matched.T, smoothness=self.smoothness_alpha)
            loss = loss + self.smoothness_alpha * (smooth_i2t_loss['loss'] + smooth_t2i_loss['loss'])
            loss_dict['loss/loss'] = loss
            loss_dict['loss/n_pseudo_gts'] = smooth_i2t_loss['n_pseudo_gts'] + smooth_t2i_loss['n_pseudo_gts']

        return loss['loss']


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class PCMN(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        
        self.bceloss = nn.BCEWithLogitsLoss()

        self.mean_linear_image = nn.Linear(self.embed_dim, self.embed_dim)
        self.mean_linear_text = nn.Linear(self.embed_dim, self.embed_dim)

        self.mean_ln_image = nn.LayerNorm(self.embed_dim)
        self.mean_ln_text = nn.LayerNorm(self.embed_dim)


        self.std_lineari = nn.Linear(512, 512)
        self.image_std_ln = nn.Identity()
        self.std_lineart = nn.Linear(512, 512)

        self.std_linearim = nn.Linear(512, 512)
        self.mask_image_std_ln = nn.Identity()
        self.std_lineartm = nn.Linear(512, 512) 

        self.utc_loss = UncertainLoss()
        
        init_shift = 5
        init_negative_scale = 5
        self.smoothness_alpha = 0.1
        shift = init_shift * torch.ones(1)
        negative_scale = init_negative_scale * torch.ones(1)
        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)
        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.inclusion_loss = InclusionLoss()

 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 

        ids = batch['pids']
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        
        idx = ids.view(-1, 1)
        g_idx = ids.view(1, -1)
        pos_idx = torch.eq(idx, g_idx).float()
        matched = pos_idx / pos_idx.sum(1, keepdim=True)

        i_cls = image_feats[:, 0, :] 
        t_cls = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1), :] 
        i_feats = torch.sigmoid(i_cls)
        i_feats = self.mean_linear_image(i_feats)
        i_feats = self.mean_ln_image(i_feats)
        i_feats = F.normalize(i_feats, p=2, dim=-1)
        t_feats = torch.sigmoid(t_cls)
        t_feats = self.mean_linear_text(t_feats)
        t_feats = self.mean_ln_text(t_feats)
        t_feats = F.normalize(t_feats, p=2, dim=-1)        
        image_feats_std = self.std_lineari(image_feats)[:, 0, :]
        image_feats_std = self.image_std_ln(image_feats_std)
        text_feats_std = self.std_lineart(text_feats)
        text_feats_std = text_feats_std[torch.arange(text_feats_std.shape[0]), caption_ids.argmax(dim=-1)].float()

        loss_fn = self._compute_prob_matching_loss
        loss_dict = loss_fn(i_feats, t_feats, image_feats_std, text_feats_std, matched) 
        loss = loss_dict['loss'] + loss_fn(t_feats, i_feats, text_feats_std, image_feats_std, matched)['loss']
        smooth_i2t_loss = loss_fn(i_feats, t_feats, image_feats_std, text_feats_std, \
                                 matched=matched, smoothness=self.smoothness_alpha) 
        smooth_t2i_loss = loss_fn(t_feats, i_feats, text_feats_std, image_feats_std, \
                                 matched=matched.T, smoothness=self.smoothness_alpha) 
        loss = loss + self.smoothness_alpha * (smooth_i2t_loss['loss'] + smooth_t2i_loss['loss'])

        image_all = {"mean": i_feats, "std": image_feats_std}
        text_all = {"mean": t_feats, "std": text_feats_std}


        mask_caption_ids = batch['mask_caption_ids']
        mask_image_feats, atten_i, mask_text_feats, atten_t = self.base_model(images, mask_caption_ids, 0.75)
        mask_i_feats = mask_image_feats[:, 0, :].float()
        mask_image_feats_std = self.std_linearim(mask_image_feats)[:, 0, :]
        mask_image_feats_std = self.mask_image_std_ln(mask_image_feats_std)
        mask_image_all = {"mean": mask_i_feats, "std": mask_image_feats_std}

        mask_t_feats = mask_text_feats[torch.arange(mask_text_feats.shape[0]), mask_caption_ids.argmax(dim=-1)].float()
        mask_text_feats_std = self.std_lineartm(mask_text_feats)
        mask_text_feats_std = mask_text_feats_std[torch.arange(mask_text_feats_std.shape[0]), mask_caption_ids.argmax(dim=-1)].float()
        mask_text_all = {"mean": mask_t_feats, "std": mask_text_feats_std}  

        include_loss = self.inclusion_loss(image_all, text_all, masked_image_features = mask_image_all, masked_text_features = mask_text_all)

        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})

        ret.update({'uct_loss':loss})
        ret.update({'include_loss':include_loss})
  
        return ret
    
    def _compute_prob_matching_loss(self, i_feats, t_feats, image_feats_std, text_feats_std, \
                                    matched, smoothness=0):

        mu_pdist = ((i_feats.unsqueeze(1) - t_feats.unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = ((torch.exp(image_feats_std).unsqueeze(1) + torch.exp(text_feats_std).unsqueeze(0))).sum(-1)
        logits = mu_pdist + sigma_pdist
        logits = -self.negative_scale * logits + self.shift
        matched, n_pseudo_gts = self._recompute_matched(matched, logits, smoothness)
        loss = self.bceloss(logits, matched)

        return {
            'loss': loss,
            'n_pseudo_gts': n_pseudo_gts,
        }
    
    def _recompute_matched(self, matched, logits, smoothness=0):
        """ Recompute the `matched` matrix if the smoothness value is given.
        """
        if not smoothness:
            return matched, None
        else:
            logits = logits.view(matched.size())
            # XXX Warning: all negative pairs will return weird results
            gt_labels, gt_indices = torch.max(matched, dim=1)
            gt_vals = logits[:, gt_indices].diag()
            pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1))
            new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices))
            _matched = matched.clone()
            _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices]

            return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)


def build_model(args, num_classes=11003):
    model = PCMN(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
