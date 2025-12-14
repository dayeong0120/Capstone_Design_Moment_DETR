# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from moment_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from moment_detr.matcher import build_matcher
from moment_detr.transformer import build_transformer
from moment_detr.position_encoding import build_position_encoding
from moment_detr.misc import accuracy

import moment_detr.logging_state as LOG


class MomentDETR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)  # (bsz, L_vid)

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1,
                # [추가] IoU top-k auxiliary span loss 관련 하이퍼파라미터
                 topk_iou_aux=1,          # 각 GT당 추가로 잡을 query 개수 (k)
                 topk_iou_thresh=0.6,     # 이 IoU 이상인 query만 aux supervision
                 topk_iou_coef=0.15):      # 기존 span/giou loss에 섞는 비율):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        
        # [추가] IoU top-k auxiliary span loss 설정
        self.topk_iou_aux = topk_iou_aux
        self.topk_iou_thresh = topk_iou_thresh
        self.topk_iou_coef = topk_iou_coef

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    # [추가] IoU top-k auxiliary span loss 함수
    def _compute_topk_aux_span_loss(self, outputs, targets, indices):
        """
        Hungarian main match 외에, 각 GT에 대해
        IoU가 높은 query top-k에도 약한 span / giou loss를 주는 함수.

        outputs: dict, "pred_spans": [B, Q, 2] (cx, w)
        targets: list of dicts (이미 targets["span_labels"]가 들어온 상태)
        indices: list of (src_idx, tgt_idx) from matcher

        return: (aux_l1, aux_giou) 두개의 loss 텐서 
        """

        # 설정에 따라 아예 끄는 경우
        if self.topk_iou_aux <= 0 or self.span_loss_type != "l1":
            # outputs["pred_spans"]와 같은 device, 같은 dtype을 가진 스칼라 0 텐서 생성
            zero = outputs["pred_spans"].new_tensor(0.) 
            # aux loss를 안쓰겠다는 의미로 둘 다 0인 텐서를 돌려줌 
            return zero, zero
        
        pred_spans = outputs["pred_spans"]          # [B, Q, 2]
        # DETR 스타일로 전체 GT 개수로 normalization하기 위한 변수 
        # num_boxes : 배치 전체에 있는 GT 스팬의 총 개수 
        num_boxes = sum(len(t["spans"]) for t in targets)
        if num_boxes == 0:
            zero = pred_spans.new_tensor(0.)
            return zero, zero
        num_boxes = float(num_boxes) 

        # 보조 loss들을 다 더해서 모아둘 누적 변수 (초기값 0)
        total_l1 = pred_spans.new_tensor(0.)
        total_giou = pred_spans.new_tensor(0.)

        # 배치 차원(B)에 대해 하나씩 처리
        # indices는 길이 B인 리스트, 각 원소는 (src_idx, tgt_idx) 튜플
        #   src_idx: 이 샘플에서 매칭된 query 인덱스들
        #   tgt_idx: 그 query들이 매칭된 GT 인덱스들
        for b, (src_idx, tgt_idx) in enumerate(indices):
            # b번째 샘플(영상)의 GT span들: [G, 2] (cx, w)
            tgt_spans_b = targets[b]["spans"]
            # 이 샘플에 GT가 없으면 스킵
            if tgt_spans_b.numel() == 0:
                continue

            # b번째 샘플의 예측 span들: [Q, 2] (cx, w)
            pred_spans_b = pred_spans[b]
            # G: 이 샘플의 GT 개수
            G = tgt_spans_b.shape[0]

            # 예측 span과 GT span을 (start, end) 형식으로 변환
            # span_cxw_to_xx: (cx, w) → (start, end)
            pred_xx = span_cxw_to_xx(pred_spans_b)   # [Q, 2]
            tgt_xx = span_cxw_to_xx(tgt_spans_b)     # [G, 2]

            # generalized_temporal_iou:
            #   입력: [Q, 2], [G, 2]
            #   출력: [Q, G]  (각 query-각 GT 쌍의 GIoU 값)
            # 각 쿼리와 GT사이의 GIoU값을 계산한 행렬 
            iou_mat = generalized_temporal_iou(pred_xx, tgt_xx)  # [Q, G] 

            # Hungarian 결과로부터
            #   "각 GT 인덱스 gi → 그 GT에 매칭된 query 인덱스 qi" 매핑 딕셔너리 생성
            # src_idx, tgt_idx 텐서를 list로 바꿔서 zip으로 묶어줌
            main_for_gt = {
                int(gi): int(qi)
                for qi, gi in zip(src_idx.tolist(), tgt_idx.tolist())
            }

            # 이미 다른 GT에 매칭된 query들의 집합 (aux 후보에서 제외할 용도)
            matched_queries = set(main_for_gt.values())

            # 이 샘플의 모든 GT(gi=0..G-1)에 대해 반복
            for gi in range(G):
                # 이 GT가 Hungarian에서 아예 매칭 안 됐으면 (드문 케이스) 보조 loss도 줄 수 없으니 스킵
                if gi not in main_for_gt:
                    continue

                # 이 GT에 main match된 query 인덱스
                q_main = main_for_gt[gi]

                # 이 GT 컬럼에 대한 IoU 값: [Q]  (각 query가 이 GT와 가지는 IoU)
                iou_col = iou_mat[:, gi].clone() # 특정 GT gi에 대해, 모든 query의 IoU를 가져오는 것

                # main matched query는 보조 후보에서 제외 (이미 strong supervision 받는 애니까)
                iou_col[q_main] = -1.0

                # (옵션) 다른 GT에 이미 매칭된 query들도 aux 후보에서 제외
                #   → 하나의 query가 여러 GT의 aux positive가 되는 걸 막기 위함
                for q_other in matched_queries:
                    if q_other != q_main:
                        iou_col[q_other] = -1.0

                # IoU가 threshold보다 큰 query만 보조 후보로 사용
                valid_mask = iou_col > self.topk_iou_thresh
                # threshold 이상인 후보가 하나도 없으면 이 GT는 스킵
                if valid_mask.sum() == 0:
                    continue

                # valid_mask가 True인 인덱스들만 모은다: [M]
                # 즉, IoU가 threshold이상인 쿼리의 인덱스만 저장 
                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

                # 그 중에서 IoU 값이 큰 순서대로 top-k 개 선택
                # 뽑을 수 있는 만큼만 뽑고, 더는 안 뽑는다.
                k = min(self.topk_iou_aux, int(valid_idx.numel()))
                # valid한 애들 중에서만 topk를 뽑기 위해 iou_col[valid_idx] 사용
                #  iou_col[valid_idx] 는 valid_idx 쿼리들의 iou값을 배열로 저장 
                # topk_vals는 실제 top-k의 IoU값 배열, topk_pos는 top-k쿼리의 인덱스 
                topk_vals, topk_pos = torch.topk(iou_col[valid_idx], k)
                # topk_pos는 valid_idx의 인덱스이므로, 실제 쿼리 인덱스로 바꿔주기 
                """
                valid_idx = [0, 2, 4]
                topk_pos  = [0, 1]

                aux_q_idx = [ valid_idx[0], valid_idx[1] ]
                aux_q_idx = [ 0, 2 ]
                """
                aux_q_idx = valid_idx[topk_pos]  # [k']

                # 혹시라도 k가 0이 되는 경우 방어
                if aux_q_idx.numel() == 0:
                    continue

                # 보조 supervision을 줄 예측 span들: [k', 2] (cx, w)
                aux_pred = pred_spans_b[aux_q_idx]
                # 이 GT span을 aux_pred 개수만큼 복제: [k', 2]
                gt_span = tgt_spans_b[gi].unsqueeze(0).expand_as(aux_pred)

                # L1 보조 loss: (sum으로 누적, 나중에 전체 GT 개수로 나눠서 평균)
                l1 = F.l1_loss(aux_pred, gt_span, reduction="sum")

                # GIoU 보조 loss 계산
                #   1) (cx, w) → (start, end)로 변환
                aux_xx = span_cxw_to_xx(aux_pred)      # [k', 2]
                gt_xx_rep = span_cxw_to_xx(gt_span)    # [k', 2]
                #   2) generalized_temporal_iou([k',2], [k',2]) → [k', k'] 행렬이 나오므로
                #      각 쌍의 대각(diagonal)만 뽑아서 1:1 매칭으로 본다
                giou_vec = generalized_temporal_iou(aux_xx, gt_xx_rep).diag()  # [k']
                #   3) GIoU loss = 1 - GIoU
                giou_loss = (1.0 - giou_vec).sum()

                # 배치 전체 보조 loss에 누적
                total_l1 += l1
                total_giou += giou_loss

        # 배치 전체 GT 개수로 나눠서 평균 loss로 스케일 맞추기
        total_l1 = total_l1 / num_boxes
        total_giou = total_giou / num_boxes

        # aux L1 loss와 aux GIoU loss 반환
        return total_l1, total_giou


    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none') # weighted L1 loss ? 
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()

        # [추가] IoU top-k auxiliary span loss
        #  - span_loss_type == "l1" 일 때만 의미 있음
        #  - self.topk_iou_aux <= 0 이면 _compute_topk_aux_span_loss 안에서 그냥 0 리턴
        if self.span_loss_type == "l1" and self.topk_iou_aux > 0:
            # 여기서의 targets는 이미 targets["span_labels"]로 바뀐 상태 (위에서 한 줄)
            aux_l1, aux_giou = self._compute_topk_aux_span_loss(outputs, targets, indices)
            losses['loss_span'] = losses['loss_span'] + self.topk_iou_coef * aux_l1
            losses['loss_giou'] = losses['loss_giou'] + self.topk_iou_coef * aux_giou

        
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        # ㄴ 매칭된 쿼리 = foreground, 나머지 쿼리 = background로 처리
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        # ㄴ _get_src_permutation_idx(indices)는 indices에서 예측 쪽 인덱스만 뽑아서,
        # 두 개의 1D 텐서 (batch_idx, src_idx)로 바꿔줌 
            # batch_idx: 이 매칭이 어느 배치 샘플에 속하는지 (예: [0,0,1,1,1,...])
            # src_idx: 그 배치 샘플 안에서 몇 번째 query인지 (예: [3,7,0,5,...])
        idx = self._get_src_permutation_idx(indices)
        # idx의 길이는 배치 전체에서 GT 개수의 총합 = FG 개수 
        # 그 크기 그대로 모든 값을 background_label(=1)로 채운 텐서를 만듦.
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        # idx가 (batch_idx, src_idx) 형태
        # target_classes[batch_idx[k], src_idx[k]]에 해당하는 위치들이 선택
        # 그 위치들을 foreground_label(=0)으로 바꿈 
        target_classes[idx] = self.foreground_label

        # 실제 cross-entropy 분류 loss 계산 
        """ 자세한 설명 
            src_logits.transpose(1, 2) : src_logits는 (B, Q, C)인데, F.cross_entropy는 (B, #classes, #queries) 같은 형태를 기대하므로
            target_classes는 (batch_size, #queries), 각 위치에 0 또는 1이 들어있음.
            self.empty_weight: 클래스별 weight 텐서 (크기 2)
                empty_weight = torch.ones(2)
                empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
                => background 클래스의 손실 비중을 eos_coef만큼 줄여서 배경이 너무 많을 때 FG/BG 불균형 문제를 완화.
            reduction="none": CE에서 평균을 아직 내지 말고, 원소별 loss 텐서를 그대로 반환시키는 옵션
        """
        # loss_ce 형태는 (batch_size, #queries)
        """ loss_ce 예시
            [
                [0.3, 1.2, 0.8, 0.01, 0.5],   # batch 0의 각 query 손실
                [0.6, 0.1, 0.9, 0.4, 0.2],    # batch 1의 각 query 손실
            ]
        """
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        # 각 query의 loss를 하나의 스칼라로 평균
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets)

        # ---------------------- [추가] 쿼리 인덱스 히스토그램 누적 ------------------
        # 첫 배치에서 쿼리 개수(num_queries)를 동적으로 결정
        if LOG.is_training_phase:
            if LOG.matching_hist is None:
                # outputs의 query dimension을 이용해서 쿼리 개수 추출
                # pred_spans: [batch_size, num_queries, 2] 같은 형태라고 가정
                num_queries = outputs["pred_spans"].shape[1]

                # 각 쿼리 인덱스(0 ~ num_queries-1)가 몇 번 매칭되는지 셀 히스토그램
                LOG.matching_hist = torch.zeros(num_queries, dtype=torch.long)

            # indices는 보통 배치 크기만큼의 리스트이고,
            # 각 원소는 (idx_pred, idx_gt) 형태의 튜플
            #   idx_pred: 이 배치에서 GT와 매칭된 query index들 (1D tensor)
            #   idx_gt  : 해당하는 GT index들 (여긴 안 써도 됨)
            for (idx_pred, idx_gt) in indices:
                # idx_pred 안에 들어있는 각 query index를 순회하면서
                # 해당 쿼리가 매칭된 횟수를 +1 해준다.
                for q in idx_pred: # idx_pred = tensor([3, 7])
                    q_idx = int(q.item())   # tensor형태로 되어있는 query 인덱스를  → 파이썬 int로 변환
                    LOG.matching_hist[q_idx] += 1
        # ---------------------------------------------------------------------
        
        # 이번 배치의 매칭 정보 저장 (delta FG score, span 등 확인하기 위함 )
        if LOG.is_training_phase:
            LOG.CURR_MATCH = []
            for (idx_pred, idx_gt) in indices:
                LOG.CURR_MATCH.append((
                    idx_pred.detach().cpu(),
                    idx_gt.detach().cpu()
                ))

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # 보조 손실 (auxiliary loss)를 계산하는 부분 
        # decoder의 각 layer가 하나의 예측을 내기 때문에, 각 layer마다 loss를 계산해서 총 loss에 더해주는 방식
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs: # outputs['aux_outputs'] 에는 각 decoder layer의 예측 결과(classification / span 등)
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # Hungarian matcher를 각 layer 예측에 대해 따로 수행
                indices = self.matcher(aux_outputs, targets) 

                # ---------------------- [추가] aux layer 쿼리 매칭 히스토그램 ----------------------
                if LOG.is_training_phase:
                    if LOG.matching_hist_aux is None:
                        LOG.matching_hist_aux = []

                    # 현재 layer(i)의 기록 공간이 없으면 생성
                    if len(LOG.matching_hist_aux) <= i:
                        num_queries = aux_outputs["pred_spans"].shape[1]
                        LOG.matching_hist_aux.append(torch.zeros(num_queries, dtype=torch.long))

                    # 매칭된 query index 기록
                    for (idx_pred, idx_gt) in indices:
                        for q in idx_pred:
                            q_idx = int(q.item())
                            LOG.matching_hist_aux[i][q_idx] += 1
                # -------------------------------------------------------------------------

                for loss in self.losses: # 모델이 계산할 loss 목록
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    # 각 loss 타입(예: classification, span, giou 등)에 따라 loss 계산.
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    # aux layer의 loss 이름 뒤에 _i 붙이기 ex) layer 1: loss_class_0
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict) # 최종 loss dict에 aux layer loss를 추가

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
         # [추가] IoU top-k aux span loss 하이퍼파라미터
        topk_iou_aux=1,          # 각 GT당 IoU 높은 query 2개까지 보조 supervision
        topk_iou_thresh=0.6,     # IoU 0.5 이상만 대상
        topk_iou_coef=0.15        # 기존 loss에 0.3 비율로 섞기
    )
    criterion.to(device)
    return model, criterion
