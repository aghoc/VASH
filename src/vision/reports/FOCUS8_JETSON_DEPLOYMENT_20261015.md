# 요청 8종 개선 모델의 Jetson 적용 기록

2026-10-15 KST.

기존 카메라 서비스가 사용하던 캔·동전 보강 모델과 직전 8종 추가학습 모델을 같은 Jetson에서 비교했다. 직전 추가학습 checkpoint가 Jetson에 없었던 것을 확인한 뒤 전송·native TensorRT 변환·검증했다.

배포 전 조건 통과: **True**. 서비스에 새 모델 적용: **True**.

이번 별도 집중 실험(v7)의 5개 후보는 validation 퇴행으로 전부 미채택했다. 여기서 비교·적용하는 모델은 이전에 GPU 학습과 독립 test를 마친 `yolo26s_focus8_v5.pt`다. [집중 실험 결과](FOCUS8_TARGETED_RESULT_20261015.md)와 [이전 독립 test](FOCUS8_FINETUNE_RESULT_20261015.md)를 함께 참고한다.

## 같은 Jetson에서의 정확도 비교

두 engine 모두 FP16, static 640px, batch 1. 아래 수치는 validation AP50–95×100이며 정답률이 아니다. engine의 정사각형 전처리와 FP16 때문에 이전 RTX FP32 수치와 직접 빼서 비교하지 않는다. 새 test 평가나 카메라 영상 접근·정답 수집은 하지 않았다.

| 검증 영역 | 사진 | 대상 클래스 | 기존 평균 AP | 새 평균 AP |
| --- | ---: | ---: | ---: | ---: |
| frozen | 6,560 | 8 | 28.04 | 28.57 |
| coco | 136 | 7 | 24.36 | 24.82 |
| openimages | 248 | 8 | 24.24 | 32.87 |

평균이 개선되어도 모든 물체가 모든 영역에서 좋아진 것은 아니다. frozen 리모컨·숟가락·포크는 소폭 하락했고, COCO 칫솔은 55.01→45.65로 하락했다. Open Images 칫솔은 3.78→17.89로 개선됐지만 아직 약하다. COCO 칫솔 validation의 정답은 3개뿐이라 변화의 불확실성이 크다. 검증 영역별 범위와 정답 box 수를 함께 판단해야 한다.

### frozen의 요청 물체별 AP

| 물체 | 기존 AP | 새 AP | 변화(pp) | 정답 box |
| --- | ---: | ---: | ---: | ---: |
| 병 | 35.49 | 36.33 | +0.85 | 2698 |
| 컵 | 30.07 | 31.43 | +1.36 | 1818 |
| 휴대폰 | 25.88 | 27.51 | +1.64 | 983 |
| 리모컨 | 34.88 | 34.36 | -0.52 | 690 |
| 숟가락 | 26.63 | 26.28 | -0.35 | 799 |
| 포크 | 45.24 | 44.37 | -0.87 | 294 |
| 칫솔 | 7.95 | 8.99 | +1.04 | 84 |
| 펜 | 18.20 | 19.26 | +1.06 | 1256 |

### coco의 요청 물체별 AP

| 물체 | 기존 AP | 새 AP | 변화(pp) | 정답 box |
| --- | ---: | ---: | ---: | ---: |
| 병 | 25.86 | 27.93 | +2.07 | 83 |
| 컵 | 22.05 | 24.88 | +2.83 | 114 |
| 휴대폰 | 10.79 | 10.87 | +0.09 | 34 |
| 리모컨 | 16.09 | 17.52 | +1.44 | 33 |
| 숟가락 | 11.15 | 12.86 | +1.71 | 33 |
| 포크 | 29.55 | 34.04 | +4.48 | 29 |
| 칫솔 | 55.01 | 45.65 | -9.35 | 3 |

### openimages의 요청 물체별 AP

| 물체 | 기존 AP | 새 AP | 변화(pp) | 정답 box |
| --- | ---: | ---: | ---: | ---: |
| 병 | 34.96 | 39.96 | +5.00 | 86 |
| 컵 | 37.46 | 48.97 | +11.50 | 40 |
| 휴대폰 | 32.37 | 47.71 | +15.34 | 62 |
| 리모컨 | 25.21 | 29.40 | +4.19 | 8 |
| 숟가락 | 26.61 | 32.78 | +6.16 | 63 |
| 포크 | 22.24 | 22.50 | +0.25 | 43 |
| 칫솔 | 3.78 | 17.89 | +14.10 | 13 |
| 펜 | 11.30 | 23.79 | +12.48 | 41 |

### 캔·동전 공개 validation 유지

367장, 별도 평가.

| 물체 | 기존 AP | 새 AP |
| --- | ---: | ---: |
| 캔 | 65.26 | 65.58 |
| 동전 | 66.54 | 67.58 |

## 처리 속도와 서비스 확인

공개 이미지 24장, warmup 20회 후 200회 측정. 전체 detection→target→grip→UART 직렬화를 포함하며 실제 UART 송신은 하지 않았다. 클럭은 고정하지 않았으며 카메라 입력·JPEG·브라우저 지연은 이 benchmark에 포함하지 않는다.

| 모델 | pipeline FPS | pipeline p95(ms) |
| --- | ---: | ---: |
| 기존 | 31.06 | 36.31 |
| 새 모델 | 27.45 | 40.70 |

서비스 active, 자동 재시작 0회. 재시작 후 처리 326프레임, 읽기 실패 0회, 상태 JSON의 처리 FPS 20.34, pipeline p95 28.47ms. 확인은 처리 통계만 사용했다. 실시간 물체 인식 정확도를 사람이 확인한 결과가 아니다.

Live viewer는 최신 JPEG 한 장만 RAM에서 교체한다. 프레임·영상을 자동 저장하지 않으며 이번 작업에서도 카메라 이미지는 조회하거나 저장하지 않았다.

## 산출물과 복구

- checkpoint SHA-256: `c60b573217426a0f600cbc7d2dea9fc67a9708ce53e4ce46fbc7248d9c794439`
- engine SHA-256: `748ec009527b76679afab7a9e875c2099aad9feb916c6337cfb981c984d9f26a`
- 새 engine: Jetson `/mnt/data/vagt-orca-ai/artifacts/deployment/tensorrt/yolo26s_focus8_v5_fp16.engine`
- 이전 engine: `yolo26s_can_coin_retention_fp16.engine` 그대로 보존.
- 검증 원본: `runs/focus8_targeted_20261015/jetson_comparison.json`, `jetson_acceptance.json`, `deployment_result.json`.
- 원래 설정/서비스의 byte 백업: Jetson `runs/focus8_targeted_20261015/predeploy_files/manifest.json`.

복구 시 이 manifest의 각 backup을 원래 path로 복사하고, backup이 null인 추가 drop-in은 삭제한다. `systemctl --user daemon-reload` 후 `systemctl --user restart vagt-camera.service`를 실행한다.

모든 배포 판정 항목:

- `native_fp16_end2end_export`: True
- `same_validation_settings`: True
- `all_15_classes_evaluated`: True
- `frozen_target_macro_at_least_98_percent`: True
- `each_external_domain_at_least_90_percent`: True
- `equal_domain_macro_improved`: True
- `no_material_frozen_target_drops`: True
- `other_7_classes_macro_at_least_90_percent`: True
- `can_coin_public_val_retained`: True
- `pipeline_p95_at_most_50ms`: True
- `pipeline_p95_at_most_120_percent_of_old`: True
