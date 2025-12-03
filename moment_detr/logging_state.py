
# ---- Logging buffers shared across matcher & train ----
LOG_EPOCHS = {0, 1, 2, 3, 5, 7, 10, 20, 50, 100, 150, 199}
matching_hist = None       # 각 epoch의 매칭 히스토그램
matching_hist_aux = None
is_training_phase = False  # 훈련 단계 여부
IOU_MISMATCH_BUFFER = []   # IoU 높은데 매칭 실패한 케이스 임시 저장
QUERY_MISMATCH_COUNT = None    # 쿼리별 mismatch 횟수 저장용
QUERY_FG_SCORES = None
QUERY_SPAN_LEN=None
QUERY_SPAN_CX = None

# 매칭 정보 (각 배치 마지막 forward 결과)
CURR_MATCH = None     # list[ (pred_idx, gt_idx) ] (batch 크기만큼)

# ΔFG 로깅
DELTA_FG_MATCHED = None
DELTA_FG_UNMATCHED = None

# matched일 때: GT 방향 이동 여부
TOWARDS_MATCHED = None        # per query: 0/1
DELTA_CX_MATCHED = None
DELTA_W_MATCHED  = None

# unmatched일 때: drift 패턴
CX_INC_UNMATCHED = None
CX_DEC_UNMATCHED = None
W_INC_UNMATCHED  = None
W_DEC_UNMATCHED  = None
DELTA_CX_UNMATCHED = None
DELTA_W_UNMATCHED = None

# 실제 이동 케이스
SAMPLE_SPAN_MOVES = []

# Δlogit 로깅
DELTA_LOGIT_MATCHED = None
DELTA_LOGIT_UNMATCHED = None