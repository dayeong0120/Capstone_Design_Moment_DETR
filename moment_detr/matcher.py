# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from moment_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

import moment_detr.logging_state as LOG

"""
문제 상황: 하나의 영상에서 모델이 num_queries개(보통 10개 정도)의 구간(스팬)을 예측하고, GT(정답 스팬)는 보통 그보다 적은 개수 존재

목표: “예측 스팬 vs GT 스팬” 사이의 1:1 매칭을 찾아서, 어떤 쿼리가 어떤 GT를 책임질지 정하는 것

방법: 각 (예측, GT) 쌍마다 “코스트(= 안좋은 정도)”를 계산 → 이걸 행렬로 만든 뒤 → **헝가리안 알고리즘(LSAP)**으로 전체 코스트 합이 최소가 되는 매칭을 찾음
"""

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad() # 매칭 단계에서는 gradient 계산 필요 없으므로 비활성화
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            
                모델의 예측 결과를 담은 dict.
                포함되는 키:
                    - "pred_spans": 텐서 형태 [batch_size, num_queries, 2]
                        모델이 예측한 구간(span). (cx, w) 정규화 좌표 형식.
                    - "pred_logits": 텐서 형태 [batch_size, num_queries, num_classes]
                        각 query에 대한 클래스 로짓 값. (이 query가 FG인지 BG인지”를 나타내는 softmax 이전 점수!)

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format
                길이가 배치 사이즈인 dict 또는 리스트 
                - targets["span_labels"]안에 spans라는 키의 dict가 원소 
                - "spans"의 값 : 텐서 형태 [num_target_spans, 2]
                    정답 구간(span). (cx, w) 정규화 형식.

                => 즉, 각 배치의 GT 스팬 목록 

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
            길이가 batch_size인 리스트를 반환.
            각 원소는 (index_i, index_j) 형태의 튜플이며,
                - index_i: 매칭된 "예측 query 인덱스" 리스트
                - index_j: 매칭된 "GT span 인덱스" 리스트
              두 리스트는 같은 길이를 가지며,
              길이는 min(num_queries, num_target_spans).

            즉, 각 배치마다:
                예측 스팬 중 일부를 GT 스팬과 1:1로 매칭한 결과를 반환.
                매칭되지 않은 예측 query는 BG(no-object)로 간주됨.
        """
        bs, num_queries = outputs["pred_spans"].shape[:2] # bs는 batch size 
        targets = targets["span_labels"] # 길이가 batch_size인 리스트, 각 원소는 {"spans": Tensor(...)} 형태

        # Also concat the target labels and spans

        # 각 query에 대해 softmax로 클래스 확률 계산 
        # 즉 out_prob[q] = [p_fg, p_bg]는 쿼리 q의 fg, bg일 확률 
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # ------------------------------------------
        # [추가] Query FG score 기록

        # out_prob: shape [bs*num_queries, 2]
        # reshape to [bs, num_queries, 2]
        fg_scores = out_prob.view(bs, num_queries, 2)[..., 1]   # foreground score

        # 초기화: 쿼리 개수에 맞춰 리스트 생성
        if LOG.QUERY_FG_SCORES is None:
            LOG.QUERY_FG_SCORES = [[] for _ in range(num_queries)]

        # 각 batch & query별 score 누적
        for b in range(bs):
            for q in range(num_queries):
                LOG.QUERY_FG_SCORES[q].append(float(fg_scores[0, q]))
        # ------------------------------------------
        # [추가] Query별 예측 구간의 중심값 및 길이 기록
        pred_spans = outputs["pred_spans"]  # (bs, num_queries, 2)
        span_centers  = pred_spans[..., 0]        # (bs, num_queries)
        span_lengths = pred_spans[..., 1]   # (bs, num_queries)

        # 전역 버퍼 초기화
        if LOG.QUERY_SPAN_CX is None:
            LOG.QUERY_SPAN_CX = [[] for _ in range(num_queries)]

        if LOG.QUERY_SPAN_LEN is None:
            LOG.QUERY_SPAN_LEN = [[] for _ in range(num_queries)]
        
        # 배치별로 기록
        for b in range(bs):
            for q in range(num_queries):
                cx = float(span_centers[b, q])
                length = float(span_lengths[b, q])
                LOG.QUERY_SPAN_CX[q].append(cx)
                LOG.QUERY_SPAN_LEN[q].append(length)
        # ------------------------------------------

        # 모든 배치의 GT 스팬을 하나로 이어붙임. 즉 tgt_spans은 (총 target 스팬의 개수, 2) 
        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        # 모든 GT 스팬의 클래스는 0(foreground)임. GT 스팬의 수만큼 0을 채운 리스트. tgt_ids = [0, 0, 0, ..., 0]
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]

        # classification cost 계산 부분
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class]. 
            # ㄴ 매칭에서는 1 - p(target_class)를 사용
            # ㄴ FG일 확률이 높으면 cost가 낮아야 하므로 음수를 붙인다 
        # The 1 is a constant that doesn't change the matching, it can be omitted. 
            # ㄴ 실제로는 상수 1은 빼도 순위에는 영향이 없으므로, -p(target_class)만 사용

        # out_prob[:, [0,0,0,0,0,0]] : out_prob의 모든 행을 선택하고, 열 0을 총 6번 고른다. 
        # out_prob의 행은 쿼리 하나, 열은 0이 fg일 확률, 1이 bg일 확률이므로 
        # 그림으로 보면 cost_class는
        """
        out_prob (20 × 2)의 형태 
            query0: [p_fg, p_bg]
            query1: [p_fg, p_bg]
            ...
            query19: [p_fg, p_bg]


        tgt_ids = [0,0,0,0,0,0]

        cost_class = out_prob[:, tgt_ids] 의 형태  (20 × 6)
            query0 → [p_fg, p_fg, p_fg, p_fg, p_fg, p_fg]
            query1 → [p_fg, p_fg, p_fg, p_fg, p_fg, p_fg]
            query2 → [p_fg, p_fg, p_fg, p_fg, p_fg, p_fg]...
        """

        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]


        # --------------------------------
        # 3. span cost + GIoU cost 계산
        # --------------------------------
        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            # out_spans: (bs, num_queries, 2) → (bs * num_queries, 2)
            #   각 행은 "하나의 예측 쿼리 스팬(cx, w)"를 의미
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            # L1 거리 기반 코스트
            # torch.cdist: 두 집합 사이의 pairwise 거리 계산
            # cost_span의 한 행(row)은 하나의 query가 모든 GT span과 갖는 L1 거리 벡터 (bs * num_queries, total_spans)
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            # [batch_size * num_queries, total #spans in the batch]
            # GIoU 코스트
            # span_cxw_to_xx: (cx, w) → (start, end)
            # generalized_temporal_iou: [0,1] IoU 값을 리턴 (1이 best, 0이 worst)
            # 비용이니까 -IoU 를 사용 (IoU가 클수록 코스트는 작아져야 함)
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        # L1만 쓰니까 안봐도 됨 
        else:
            # span_loss_type이 "l1"이 아닌 경우: 스팬을 "분포"로 예측했다고 가정
            # pred_spans: (bs, num_queries, max_v_l * 2)
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # pred_spans = pred_spans.repeat(1, n_spans, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, max_v_l, 2)
            # tgt_spans = tgt_spans.view(1, n_spans, 2).repeat(bs * num_queries, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, 2)
            # cost_span = pred_spans[tgt_spans]
            # cost_span = cost_span.view(bs * num_queries, n_spans)

            # giou
            cost_giou = 0

        # --------------------------------
        # 4. 최종 cost matrix 구성
        # --------------------------------
        # import ipdb; ipdb.set_trace()
        # 최종 코스트 = 가중치 * (각 코스트 항목)
        # 각 항목의 shape는 전부 (bs * num_queries, total_spans) 이므로 그대로 더할 수 있음
        """
        ✔ cost_span
        → L1 거리 = 좌표 기반 거리
        → 값 작을수록 GT와 비슷한 위치

        ✔ cost_giou
        → IoU 기반 거리 = 실제 구간의 겹침 정도
        → IoU 높을수록 cost 낮아짐

        ✔ cost_class
        → query가 FG일 확률(=softmax 확률)을 cost에 반영
        → FG 확률 낮은 query는 자동으로 매칭에서 밀림

        span(L1): “중심과 너비 값이 GT와 얼마나 가까운가”
        gIoU: “예측된 구간이 GT 구간을 얼마나 잘 덮는가(실제 겹침)”
        """
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        
        # 하나의 cost matrix를 다시 영상 단위로 분리하는 과정 
        # C: (bs * num_queries, total_spans) → (bs, num_queries, total_spans)
        C = C.view(bs, num_queries, -1).cpu() # 헝가리안 알고리즘은 numpy 기반이라 CPU로 옮김

        sizes = [len(v["spans"]) for v in targets]

        """
        C.split(...) 결과 예시 (bs=3)
            c0: (3, 10, 2)   # video0 GT span 2개
            c1: (3, 10, 4)   # video1 GT span 4개
            c2: (3, 10, 1)   # video2 GT span 1개

        enumerate를 쓰면:
            i=0, c=c0
            i=1, c=c1
            i=2, c=c2
        즉 각 영상마다의 cost matrix를 헝가리안 알고리즘에 넣는다 

        indices : (query_idx_list, gt_idx_list)의 배열. 
            indices = [
                (array([0, 3]), array([1, 0])),       # video0
                (array([2, 7, 9]), array([0, 1, 2]))  # video1
            ]  
            video0의 query0는 GT span1로 매칭
            video0의 query3는 GT span0으로 매칭 
        """
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # ======== [추가: predicted spans 계산] ========
        # outputs["pred_spans"] shape = (bs, num_queries, 2) = [cx, w]
        pred_cx = outputs["pred_spans"][..., 0]  # (bs, num_queries)
        pred_w  = outputs["pred_spans"][..., 1]  # (bs, num_queries)

        # convert to start/end
        pred_start = pred_cx - pred_w / 2
        pred_end   = pred_cx + pred_w / 2
        # ============================================

        # ------------------------------------------
        # [추가] IoU 높지만 매칭되지 않은 query 기록
        IOU_THRESH = 0.5

        iou_mismatch_list = []  # 이번 매칭 결과에서 발견된 mismatch를 저장할 리스트
        # 쿼리 mismatch 카운트 초기화
        if LOG.QUERY_MISMATCH_COUNT is None:
            LOG.QUERY_MISMATCH_COUNT = [0 for _ in range(num_queries)]

        # cost_giou는 "-IoU" 형태의 값이므로 반대로 부호를 바꾸면 IoU 값이 됨
        # cost_giou shape: (bs * num_queries, total_spans)
        # reshape해서 batch 단위로 보기 쉽게 만들기
        giou_mat = -cost_giou.view(bs, num_queries, -1)  
        # 즉 giou_mat[b][q][gi] = batch b, query q, GT gi 사이의 IoU 값

        # 각 cost도 batch 단위로 보기 쉽게 reshape (비교/기록용)
        cost_class_b = cost_class.view(bs, num_queries, -1)
        cost_span_b  = cost_span.view(bs, num_queries, -1)
        cost_giou_b  = cost_giou.view(bs, num_queries, -1)
        C_b          = C  # 최종 cost matrix는 이미 (bs, num_queries, total_spans) 형태

        prob_3d = outputs["pred_logits"].softmax(-1) # 로깅시에는 out_prob대신 사용 [bs, Q, C]

        # 각 배치마다 반복
        for b, (pred_idx, tgt_idx) in enumerate(indices):

            # GT gi → matched query q_matched 매핑
            matched_query_for_gt = {
                gi.item(): q.item() for q, gi in zip(pred_idx, tgt_idx)
            }

            # 이번 배치의 GT 개수
            num_gt = len(targets[b]["spans"])
            #  batch b의 GT column 시작
            start_col = sum(sizes[:b])

            # GT 기준으로 mismatch 탐지
            for gi in range(num_gt):
            
                # 이 GT에 실제로 매칭된 query
                q_matched = matched_query_for_gt.get(gi, None)
                if q_matched is None:
                    continue
                
                # 현재 GT의 열 인덱스 
                col = start_col + gi
                
                # 매칭된 쿼리의 IoU
                iou_matched = float(giou_mat[b, q_matched, col])

                # 모든 query에 대해 검사
                for q in range(num_queries):
                    # 이미 매칭된 query는 mismatch 후보 아님 → skip
                    if q == q_matched:
                        continue

                    # 이 query가 다른 GT에 매칭된 경우 skip
                    if q in matched_query_for_gt.values():
                        continue

                    iou_q = float(giou_mat[b, q, col])

                    # mismatch 조건
                    if iou_q > iou_matched + 0.1:
                        LOG.QUERY_MISMATCH_COUNT[q] += 1 

                        # 관심 쿼리만 상세 기록
                        if LOG.WIDE_QUERY_FINAL is not None and q == LOG.WIDE_QUERY_FINAL:
                            # 정렬을 위해 미리 필요한 diff 값들 계산
                            iou_diff  = iou_q - iou_matched
                            fg_diff   = float(prob_3d[b, q, 1]) - float(prob_3d[b, q_matched, 1])
                            cost_diff = float(C_b[b, q, col]) - float(C_b[b, q_matched, col])

                            # 상세 cost breakdown 포함해서 기록
                            iou_mismatch_list.append({
                                "batch": b, # 배치 index
                                "query": q, # 매칭 실패한 query index
                                "matched_query_index": int(q_matched),

                                "gt": col, # IoU가 높은 GT index
                                "iou_q": iou_q, # IoU 값
                                "iou_matched": iou_matched,
                                "iou_diff": iou_diff,     # ⭐ 정렬용 필드 1
                                
                                "class_cost_q":       float(cost_class_b[b, q, col]),
                                "class_cost_matched": float(cost_class_b[b, q_matched, col]),

                                "l1_cost_q":          float(cost_span_b[b, q, col]),
                                "l1_cost_matched":    float(cost_span_b[b, q_matched, col]),

                                "giou_cost_q":        float(cost_giou_b[b, q, col]),
                                "giou_cost_matched":  float(cost_giou_b[b, q_matched, col]),

                                "final_cost_q":       float(C_b[b, q, col]),
                                "final_cost_matched": float(C_b[b, q_matched, col]),
                                "cost_diff": cost_diff,   # ⭐ 정렬용 필드 2
                                # ===== predicted span (this query) =====
                                "pred_span_q": [
                                    float(pred_start[b, q]),
                                    float(pred_end[b, q])
                                ],
                                "pred_cx_q": float(pred_cx[b, q]),
                                "pred_w_q":  float(pred_w[b, q]),

                                # ===== matched query의 predicted span =====
                                "pred_span_matched": [
                                    float(pred_start[b, q_matched]),
                                    float(pred_end[b, q_matched])
                                ],
                                "pred_cx_matched": float(pred_cx[b, q_matched]),
                                "pred_w_matched":  float(pred_w[b, q_matched]),

                                # ===== GT span =====
                                "gt_span": [
                                    float(targets[b]["spans"][gi][0]),
                                    float(targets[b]["spans"][gi][1])
                                ],

                                # FG score 비교
                                "fg_score_q": float(prob_3d[b, q, 1]),  # Query 0의 FG score
                                "fg_score_matched": float(prob_3d[b, q_matched, 1]),  # matched query의 FG score
                                "fg_diff": fg_diff
                            })

            # Option 1: 전역 리스트에 저장 (추천)
            LOG.IOU_MISMATCH_BUFFER.extend(iou_mismatch_list)
        # ------------------------------------------

        # 이 numpy 배열들을 torch 텐서로 감싸서 반환
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )
