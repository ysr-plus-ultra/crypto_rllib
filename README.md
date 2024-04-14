# crypto_rllib

- main.py: 모델 학습 실행

- model_serve.py: 학습된 모델 Deploy용

- rnnlib 관련: original https://github.com/daehwannam/pytorch-rnn-library

- custom/custom_model.py: pytorch 기반 모델 작성

- custom/impala_custom.py, custom/impala_torch_policy_custom.py: Impala기반 POP-ART (reward normalization) 적용

- POP-ART 적용시

rllib/algorithms/impala 안에있는
vtrace_tf.py 도 수정해야됨
line 38~
```Python
VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behaviour_action_log_probs",
        "target_action_log_probs",
        "pg_values",
        "clipped_pg_rhos"
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages pg_values clipped_pg_rhos")
```
