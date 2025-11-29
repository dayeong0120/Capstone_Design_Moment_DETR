
# ---- Logging buffers shared across matcher & train ----
LOG_EPOCHS = {0, 1, 2, 3, 5, 7, 10, 20, 50, 100, 150, 199}
matching_hist = None       # 각 epoch의 매칭 히스토그램
is_training_phase = False  # 훈련 단계 여부
IOU_MISMATCH_BUFFER = []   # IoU 높은데 매칭 실패한 케이스 임시 저장
QUERY_MISMATCH_COUNT = None    # 쿼리별 mismatch 횟수 저장용
QUERY_FG_SCORES = None
QUERY_SPAN_LEN=None
QUERY_SPAN_CX = None