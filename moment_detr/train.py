import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from moment_detr.config import BaseOptions
from moment_detr.start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from moment_detr.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters


import logging
import sys

import moment_detr.logging_state as LOG

print("[DEBUG] import moment_detr.train reached", file=sys.stderr)
sys.stderr.flush()
print(f"[DEBUG] __name__ = {__name__}")
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# 로깅 관련 요약 통계 함수 
def safe_stats(arr):
    if arr is None or len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": len(arr)
    }

def safe_ratio(arr):
    if arr is None or len(arr) == 0:
        return 0.0
    return float(sum(arr) / len(arr))


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):

    LOG.is_training_phase = True            # ← 훈련 중이라는 플래그 켜기
    LOG.matching_hist = None                # ← 매 epoch마다 초기화
    LOG.IOU_MISMATCH_BUFFER = [] 
    LOG.QUERY_MISMATCH_COUNT = None
    LOG.QUERY_FG_SCORES = None

    LOG.DELTA_FG_MATCHED = None
    LOG.DELTA_FG_UNMATCHED = None
    LOG.DELTA_LOGIT_MATCHED = None
    LOG.DELTA_LOGIT_UNMATCHED = None

    LOG.TOWARDS_MATCHED = None
    LOG.DELTA_CX_MATCHED = None
    LOG.DELTA_W_MATCHED = None
    LOG.CX_INC_UNMATCHED = None
    LOG.CX_DEC_UNMATCHED = None
    LOG.W_INC_UNMATCHED = None
    LOG.W_DEC_UNMATCHED = None
    LOG.DELTA_CX_UNMATCHED = None
    LOG.DELTA_W_UNMATCHED = None
    LOG.QUERY_SPAN_CX = None
    LOG.QUERY_SPAN_LEN = None
    LOG.SAMPLE_SPAN_MOVES = []
    logger.info(f"[Epoch {epoch_i+1}]")

    # 모델과 criterion을 학습모드로 전환 
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter) # 시간 측정을 위한 meter들을 저장할 딕셔너리
    loss_meters = defaultdict(AverageMeter)  # 손실 값들을 저장할 meter 딕셔너리

    num_training_examples = len(train_loader) # train_loader 전체 배치 개수 (tqdm 프로그레스바 total 설정에 사용)
    timer_dataloading = time.time() # 첫 배치 로딩 시작 시각 기록
    
    # train_loader에서 배치를 하나씩 꺼내면서 학습 루프 수행
    for batch_idx, batch in tqdm(
            enumerate(train_loader), # 배치 인덱스와 배치 데이터
            desc="Training Iteration", # tqdm 진행바 앞에 붙는 설명 문구
            total=num_training_examples # 전체 반복 횟수(배치 수)
        ):
        # 직전 배치부터 지금까지 걸린 dataloading 시간 측정 및 누적
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        # ----------- 입력 준비 -----------
        timer_start = time.time() # 입력 준비(전처리, GPU 이동 등) 시작 시간 기록
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start) # 입력 준비에 걸린 시간 측정 및 누적
        # --------------------------------

        # ----------- 모델 forward (모델에 입력 넣어 예측 결과 생성 및 loss 계산) -----------
        timer_start = time.time()
        outputs = model(**model_inputs)  # 모델에 입력을 넣어 예측 결과(outputs) 생성
        # criterion을 통해 outputs와 targets로부터 개별 loss 항목들을 계산 (dict 형태)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict # 각 loss에 곱해줄 가중치 딕셔너리. weight_dict 안에는 aux loss weight도 들어있다
        # 전체 손실
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)
        # -----------------------------------------------------------------------------

        # ~~~~~~~~~ [추가] ΔFG / Δlogit / ΔSPAN 로깅 ~~~~~~~~~~~
        # 1) forward 직후 baseline 기록
        fg_prob_t = outputs["pred_logits"].softmax(-1)[...,1].detach().cpu()  # FG baseline
        logit_t = outputs["pred_logits"][...,1].detach().cpu() # raw logit (더 정확한 경향 분석용)
        span_t = outputs["pred_spans"].detach().cpu()  # span baseline
        # ~~~~~~~~~~~ 이후 계속 ~~~~~~~~~~~~~
        
        # ----------- backward (역전파) -----------
        timer_start = time.time()
        optimizer.zero_grad() # 기존에 계산되어 있던 gradient 초기화
        # 전체 손실(losses)에 대해 역전파 수행 → 각 파라미터에 gradient 계산
        losses.backward()
        # gradient clip 설정이 되어 있으면, gradient norm을 opt.grad_clip 값으로 제한
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        # optimizer를 한 스텝 진행시켜 파라미터 업데이트
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)
        # ----------------------------------------

        # ~~~~~~~~~~ 2) optimizer.step() 후 같은 배치로 다시 forward ~~~~~~~~~~~
        with torch.no_grad():
            outputs_t1 = model(**model_inputs)
            fg_prob_t1 = outputs_t1["pred_logits"].softmax(-1)[...,1].cpu()
            logit_t1 = outputs_t1["pred_logits"][...,1].cpu()
            span_t1 = outputs_t1["pred_spans"].cpu()
        
        # 3) 변화량 계산
        delta_fg = fg_prob_t1 - fg_prob_t # (bs, Q)
        delta_logit = logit_t1 - logit_t
        delta_span = span_t1 - span_t # (bs, Q, 2)
        delta_cx = delta_span[...,0] # (bs, Q)    
        delta_w  = delta_span[...,1] # (bs, Q)     

        # ==== 로깅 버퍼 초기화 ====
        if LOG.DELTA_FG_MATCHED is None:
            Q = delta_fg.shape[1]
            LOG.DELTA_FG_MATCHED   = [[] for _ in range(Q)]
            LOG.DELTA_FG_UNMATCHED = [[] for _ in range(Q)]
            LOG.DELTA_LOGIT_MATCHED   = [[] for _ in range(Q)]
            LOG.DELTA_LOGIT_UNMATCHED = [[] for _ in range(Q)]

            LOG.TOWARDS_MATCHED = [[] for _ in range(Q)]
            LOG.DELTA_CX_MATCHED = [[] for _ in range(Q)]
            LOG.DELTA_W_MATCHED  = [[] for _ in range(Q)]

            LOG.CX_INC_UNMATCHED = [[] for _ in range(Q)]
            LOG.CX_DEC_UNMATCHED = [[] for _ in range(Q)]
            LOG.W_INC_UNMATCHED  = [[] for _ in range(Q)]
            LOG.W_DEC_UNMATCHED  = [[] for _ in range(Q)]

            LOG.DELTA_CX_UNMATCHED = [[] for _ in range(Q)]
            LOG.DELTA_W_UNMATCHED = [[] for _ in range(Q)]

        # ==== 현재 배치 매칭 정보 ====
        match = LOG.CURR_MATCH     # list of (pred_idx, gt_idx) for each batch
        bs = delta_fg.shape[0]
        Q = delta_fg.shape[1]

        for b in range(bs):
            # 매칭 dict 생성: q → gt_index
            batch_pred, batch_gt = match[b]
            matched_dict = { int(q): int(gi) for q, gi in zip(batch_pred.tolist(), batch_gt.tolist()) }

            for q in range(Q):
                is_matched = (q in matched_dict)

                cx0 = float(span_t[b,q,0])
                cx1 = float(span_t1[b,q,0])
                w0  = float(span_t[b,q,1])
                w1  = float(span_t1[b,q,1])

                # ==== (A) matched → GT 기준 이동 체크 ====
                if q in matched_dict:
                    gi = matched_dict[q]
                    gt_cx = float(targets["span_labels"][b]["spans"][gi][0])

                    moved_towards = abs(cx1 - gt_cx) < abs(cx0 - gt_cx) # 역전파 후 GT와의 차이가 더 작아졌는지 확인 

                    LOG.TOWARDS_MATCHED[q].append(1 if moved_towards else 0)
                    LOG.DELTA_CX_MATCHED[q].append(cx1 - cx0)
                    LOG.DELTA_W_MATCHED[q].append(w1 - w0)

                    # raw sample (matched만)
                    if len(LOG.SAMPLE_SPAN_MOVES) < 200:
                        LOG.SAMPLE_SPAN_MOVES.append({
                            "query": q,
                            "batch": b,
                            "matched": True,
                            "cx_t": cx0,
                            "cx_t1": cx1,
                            "gt_cx": gt_cx,
                            "towards": moved_towards,
                            "delta_cx": cx1 - cx0,
                            "delta_w": w1 - w0
                        })
                # ==== (B) unmatched → drift 패턴만 체크 ====
                else:
                    LOG.CX_INC_UNMATCHED[q].append(1 if cx1 > cx0 else 0)
                    LOG.CX_DEC_UNMATCHED[q].append(1 if cx1 < cx0 else 0)
                    LOG.W_INC_UNMATCHED[q].append(1 if w1 > w0 else 0)
                    LOG.W_DEC_UNMATCHED[q].append(1 if w1 < w0 else 0)
                    LOG.DELTA_CX_UNMATCHED[q].append(cx1 - cx0)
                    LOG.DELTA_W_UNMATCHED[q].append(w1 - w0)

                    # raw sample (unmatched drift)
                    if len(LOG.SAMPLE_SPAN_MOVES) < 200:
                        LOG.SAMPLE_SPAN_MOVES.append({
                            "query": q,
                            "batch": b,
                            "matched": False,
                            "cx_t": cx0,
                            "cx_t1": cx1,
                            "delta_cx": cx1 - cx0,
                            "delta_w": w1 - w0
                        })
                
                # ---- ΔFG 로깅 ----
                if is_matched:
                    LOG.DELTA_FG_MATCHED[q].append(float(delta_fg[b, q]))
                    LOG.DELTA_LOGIT_MATCHED[q].append(float(delta_logit[b, q]))
                else:
                    LOG.DELTA_FG_UNMATCHED[q].append(float(delta_fg[b, q]))
                    LOG.DELTA_LOGIT_UNMATCHED[q].append(float(delta_logit[b, q]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 전체 손실 값을 float로 변환해 logging용으로 loss_dict에 추가
        loss_dict["loss_overall"] = float(losses)  # for logging only
        # 각 loss 항목을 loss_meters에 누적해서 에폭 단위 평균을 계산하기 위함
        for k, v in loss_dict.items():
            # weight_dict에 있는 손실은 가중치가 곱해진 값을 기준으로 로깅
            # weight_dict에 없는 손실은 있는 그대로 평균 내기
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        # 다음 배치 로딩 시간 측정을 위해 현재 시각을 저장
        timer_dataloading = time.time()
        # debug 모드일 때, 3번째 배치까지만 돌리고 조기 종료 (빠른 디버깅용)
        if opt.debug and batch_idx == 3:
            break

    # ========== 한 에폭이 끝난 후, TensorBoard에 로그 기록 ==========
    # print/add logs
    # 현재 에폭의 learning rate를 TensorBoard에 기록
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    # loss_meters에 누적된 각 손실 항목의 평균을 TensorBoard에 기록
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    # 텍스트 로그 파일에 쓸 문자열 포맷 구성
    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),  # 기록 시각 문자열
        epoch=epoch_i+1, # 에폭 번호
        # "loss_name value" 형태의 문자열들을 공백으로 이어붙인 것
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()])
    )
    # 학습 로그 텍스트 파일에 append 모드로 기록
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    # ===== 시간 통계 로그 출력 =====
    logger.info("Epoch time stats:")
    # dataloading, forward, backward 각각에 대해 max/min/avg 시간 요약 출력
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")
    
    if epoch_i in LOG.LOG_EPOCHS:
        exp_dir = f"logs_hungarian/{opt.exp_id}"
        os.makedirs(exp_dir, exist_ok=True)  # 로그 저장용 폴더 생성

        # ====== (1) 메인 로그 파일 ====== (epoch 단위)
        save_path = f"{exp_dir}/epoch_debug_{epoch_i}.jsonl"
        with open(save_path, "w") as f:

            # 1) Query Matching Histogram
            if LOG.matching_hist is not None:
                f.write(json.dumps({
                    "type": "matching_hist",
                    "epoch": epoch_i,
                    "hist": LOG.matching_hist.tolist()
                }) + "\n")

            # 2) Query Mismatch Count
            if LOG.QUERY_MISMATCH_COUNT is not None:
                f.write(json.dumps({
                    "type": "query_mismatch_count",
                    "epoch": epoch_i,
                    "count": LOG.QUERY_MISMATCH_COUNT
                }) + "\n")

            # **IOU mismatch는 이제 main에 기록하지 않음**
            # # 3) IoU Mismatch Detailed Logs
            # for entry in LOG.IOU_MISMATCH_BUFFER:
            #     rec = {
            #         "type": "iou_mismatch",
            #         "epoch": epoch_i,
            #         **entry
            #     }
            #     f.write(json.dumps(rec) + "\n")

            # 4) Query FG Score 평균
            if LOG.QUERY_FG_SCORES is not None:
                fg_avg = [
                    float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0
                    for vals in LOG.QUERY_FG_SCORES
                ]

                f.write(json.dumps({
                    "type": "query_fg_avg",
                    "epoch": epoch_i,
                    "fg_avg": fg_avg
                }) + "\n")
            
            # 5) Query Norm (L2 norm of query embeddings)
            query_norms = model.query_embed.weight.norm(dim=1).detach().cpu().tolist()

            f.write(json.dumps({
                "type": "query_norms",
                "epoch": epoch_i,
                "query_norms": query_norms
            }) + "\n")

            # 6) Query별 예측구간의 중심값, 길이 평균
            f.write(json.dumps({
                "type": "query_span_stats",
                "epoch": epoch_i,
                "mean_cx":  [float(np.mean(L)) for L in LOG.QUERY_SPAN_CX],
                "std_cx":   [float(np.std(L))  for L in LOG.QUERY_SPAN_CX],
                "mean_w":   [float(np.mean(L)) for L in LOG.QUERY_SPAN_LEN],
                "std_w":    [float(np.std(L))  for L in LOG.QUERY_SPAN_LEN],
            }) + "\n")

            # 7) Aux layer 단위의 Query Matching Histograms
            if getattr(LOG, "matching_hist_aux", None) is not None:
                for layer_idx, hist_tensor in enumerate(LOG.matching_hist_aux):
                    f.write(json.dumps({
                        "type": "matching_hist_aux",
                        "epoch": epoch_i,
                        "layer": layer_idx,
                        "hist": hist_tensor.tolist()
                    }) + "\n")

            # 8) ΔFG matched/unmatched 평균
            if LOG.DELTA_FG_MATCHED is not None:
                delta_fg_matched_stats = [
                    {"query": q, **safe_stats(LOG.DELTA_FG_MATCHED[q])}
                    for q in range(Q)
                ]
                delta_fg_unmatched_stats = [
                    {"query": q, **safe_stats(LOG.DELTA_FG_UNMATCHED[q])}
                    for q in range(Q)
                ]

                f.write(json.dumps({
                    "type": "delta_fg_stats",
                    "epoch": epoch_i,
                    "matched": delta_fg_matched_stats,
                    "unmatched": delta_fg_unmatched_stats
                }) + "\n")
            
            #  9) matched일때 GT 방향으로 이동 비율
            if LOG.TOWARDS_MATCHED is not None:
                towards_ratio = [safe_ratio(vals) for vals in LOG.TOWARDS_MATCHED]

                f.write(json.dumps({
                    "type": "matched_towards_gt",
                    "epoch": epoch_i,
                    "towards_ratio": towards_ratio
                }) + "\n")
            
            # 10) matched일 때 Δcx / Δw 통계
            if LOG.DELTA_CX_MATCHED is not None:
                cx_stats = [safe_stats(vals) for vals in LOG.DELTA_CX_MATCHED]
                w_stats  = [safe_stats(vals) for vals in LOG.DELTA_W_MATCHED]

                f.write(json.dumps({
                    "type": "matched_span_change",
                    "epoch": epoch_i,
                    "delta_cx": cx_stats,
                    "delta_w": w_stats
                }) + "\n")
            
            # 11) unmatched일 때 drift 비율 및 크기
            if LOG.CX_INC_UNMATCHED is not None:
                cx_inc_ratio = [safe_ratio(vals) for vals in LOG.CX_INC_UNMATCHED]
                cx_dec_ratio = [safe_ratio(vals) for vals in LOG.CX_DEC_UNMATCHED]
                w_inc_ratio  = [safe_ratio(vals) for vals in LOG.W_INC_UNMATCHED]
                w_dec_ratio  = [safe_ratio(vals) for vals in LOG.W_DEC_UNMATCHED]

                cx_change_stats = [safe_stats(vals) for vals in LOG.DELTA_CX_UNMATCHED]
                w_change_stats  = [safe_stats(vals) for vals in LOG.DELTA_W_UNMATCHED]

                f.write(json.dumps({
                    "type": "unmatched_drift",
                    "epoch": epoch_i,
                    "cx_inc_ratio": cx_inc_ratio,
                    "cx_dec_ratio": cx_dec_ratio,
                    "w_inc_ratio": w_inc_ratio,
                    "w_dec_ratio": w_dec_ratio,
                    "delta_cx_stats": cx_change_stats,
                    "delta_w_stats": w_change_stats
                }) + "\n")

        # ====== (2) IoU mismatch 로그 파일 ======
        if LOG.IOU_MISMATCH_BUFFER is not None and len(LOG.IOU_MISMATCH_BUFFER) > 0:

            # IoU diff 기준 정렬 (넓은 쿼리 문제 분석에 가장 의의 있음)
            LOG.IOU_MISMATCH_BUFFER.sort(key=lambda x: x["iou_diff"], reverse=True)

            # top-K만 남긴다 (너가 원하는 숫자로 조절 가능)
            TOP_K = 100
            LOG.IOU_MISMATCH_BUFFER = LOG.IOU_MISMATCH_BUFFER[:TOP_K]
        
        mismatch_path = f"{exp_dir}/epoch_iou_mismatch_{epoch_i}.jsonl"
        with open(mismatch_path, "w") as f2:
            for entry in LOG.IOU_MISMATCH_BUFFER:
                rec = {
                    "type": "iou_mismatch",
                    "epoch": epoch_i,
                    **entry
                }
                f2.write(json.dumps(rec) + "\n")

        # ====== (3) Raw Span Movement Logs ======
        sample_path = f"{exp_dir}/epoch_span_moves_{epoch_i}.jsonl"
        with open(sample_path, "w") as f3:
            if LOG.SAMPLE_SPAN_MOVES is not None:
                for entry in LOG.SAMPLE_SPAN_MOVES[:200]:
                    rec = {
                        "type": "span_move_case",
                        "epoch": epoch_i,
                        **entry
                    }
                    f3.write(json.dumps(rec) + "\n")
        
        # epoch_i 는 0부터 시작한다고 가정
        if LOG.WIDE_QUERY_FINAL is None and epoch_i == 7:
            # mean_w는 matcher에서 계속 기록한 LOG.QUERY_SPAN_LEN로부터 계산 가능
            # 쿼리 개수
            num_queries = len(LOG.QUERY_SPAN_LEN)
            # epoch 7 동안 쌓인 w(mean) 계산
            mean_w = [np.mean(LOG.QUERY_SPAN_LEN[q]) for q in range(num_queries)]
            # 가장 폭이 넓은 쿼리 선택
            wide_q = int(np.argmax(mean_w))

            LOG.WIDE_QUERY_FINAL = wide_q
            print(f"[INFO] Fixed wide query at epoch 7 → {wide_q}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    print("[DEBUG] Check in stderr.", file=sys.stderr)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Val dataset size: {len(val_dataset)}")

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )
    logger.info(f"Train loader batches: {len(train_loader)}")
    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"): # 0부터 시작 
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            stop_score = metrics["brief"]["MR-full-mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def start_training():
    print("[DEBUG] alive train.start_training", flush=True) #Debugging
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio
    )

    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        eval_dataset = StartEndDataset(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    print("[DEBUG] alive2", flush=True) #Debugging
    print("[DEBUG] len(train_dataset) =", len(train_dataset), flush=True) #Debugging
    print("[DEBUG] len(eval_dataset) =", len(eval_dataset), flush=True) #Debugging
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt) #Here is the problem!!!
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug


if __name__ in ('__main__', 'moment_detr.train'):
    print("[DEBUG] alive train.main", flush=True) #Debugging
    best_ckpt_path, eval_split_name, eval_path, debug = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference()


import atexit, traceback

def goodbye():
    print("[DEBUG] program exiting", file=sys.stderr)
    traceback.print_stack(file=sys.stderr)

atexit.register(goodbye)
