# [제 4장] GPU 고속 최적화 및 자산 관리
## (Phase 4: Optimization & Play)

본 문서는 NVIDIA Isaac Lab 환경 하에서 가상 세계의 대규모 병렬 데이터 스트림을 하드웨어 비디오 메모(VRAM) 한계치까지 극대화하여 학습 대역폭을 가속하는 고속화 공정 명세서입니다. 대규모 강화학습 훈련 시 발생하는 PCIe 호스트 전송 병목을 원천 제거하는 가속 파이프라인의 핵심 개념부터, 최적 하이퍼파라미터 YAML 설정 소스코드, 실전 가동 최적화 명령어, 그리고 자산 유실을 방어하기 위한 체크포인트 롤백 및 자가 검증 매트릭스를 시스템 아키텍처 관점에서 상세히 기술합니다.

---

## **4.1 개념 이해: End-to-End GPU 가속 파이프라인 아키텍처**

전통적인 로봇 강화학습 아키텍처(예: OpenAI Gym + CPU 기반 물리 시뮬레이터)의 치명적인 한계는 시뮬레이터가 연산한 물체의 포즈 데이터를 파이썬 인터프리터를 거쳐 호스트 CPU 메모리로 다운로드한 뒤, 이를 다시 AI 신경망 훈련을 위해 GPU로 업로드하는 과정에서 발생하는 PCIe 버스 대역폭 정체였습니다. Isaac Lab은 이를 **End-to-End GPU 데이터 상주 메커니즘**으로 혁신합니다.

- **PhysX On GPU 파이프라인**: 수십~수천 개에 달하는 병렬 가상 세계의 강체 충돌, 메시 교차, 뉴턴 역학 연산이 CPU를 거치지 않고 그래픽 카드의 물리 연산 코어(PhysX) 내부에서 전부 동시에 처리됩니다.
- **리얼타임 텐서 포인터 직결**: 물리 엔진 및 관측 매니저가 산출한 24차원 관측값(Observation) 및 스테이지별 마스크 텐서는 호스트 메모리로 복사되지 않고, PyTorch 텐서(`torch.Tensor(device="cuda:0")`) 규격을 유지한 채 VRAM 주소 공간에 고정 상주합니다.
- **SKRL 배처 메모리 다이렉트 피딩**: SKRL 라이브러리의 PPO 에이전트는 VRAM 버퍼 주소의 포인터를 다이렉트로 참조하여 신경망 가중치를 업데이트(Backpropagation)하므로 데이터 전송 오버헤드가 수학적으로 0에 수렴하는 극도의 고속 드라이브를 달성합니다.

---

## **4.2 실제적인 코드 명세: 하이퍼파라미터 및 자산 관리 전략**

### **1. PPO 에이전트 설정 파일 (skrl_ppo_cfg.yaml)**
GPU 가속의 최적 연산 대역폭을 결정짓는 최적화 파라미터 파일 명세입니다. `source/franka_isaaclab/franka_isaaclab/tasks/manager_based/stack/agents/skrl_ppo_cfg.yaml` 설정 파일의 소스코드입니다.

```yaml
# SKRL PPO 고속 훈련 및 체크포인트 자산 관리 운영 스펙
seed: 42

models:
  policy:
    class: GaussianMixin
    clip_actions: True
    net: [256, 256, 128]  # 은닉층 연산 차원 최적화
  value:
    class: DeterministicMixin
    net: [256, 256, 128]

agent:
  class: PPO
  rollout_steps: 24      # 환경당 수집할 GPU 텐서 스텝 크기
  mini_batches: 4        # VRAM 대역폭 효율에 맞춘 미니배치 분할 개수
  epochs: 5              # 배치당 PPO 가중치 반복 최적화 횟수
  learning_rate: 3.0e-4  # Adam Optimizer 기본 학습률
  discount_factor: 0.99
  lambda_gae: 0.95
  clip_policy_ratio: 0.2
  entropy_loss_scale: 0.001
  value_loss_scale: 1.0
  kl_threshold: 0.015

trainer:
  class: SequentialTrainer
  timesteps: 36000                     # 총 훈련 타임스텝 제어선
  save_interval: 2000                  # 자산 백업용 롤백 마일스톤 주기 (2,000 스텝 지정)
  checkpoint_policy_with_best_name: "best_agent.pt"
  close_environment_at_exit: True
```

### **2. 태스크 및 환경 변화에 따른 하이퍼파라미터 스케일링 가이드**
태스크의 복잡도와 하드웨어 환경(예: 평행 그리퍼, 고부하 비전 센서 탑재 등)에 따라 하이퍼파라미터를 유동적으로 조율해야 OOM 크래시를 방지하고 학습 수렴을 보장할 수 있습니다.

| 파라미터 항목 | Franka 기본형 (`Template-Stack-v0`) | Doosan 심화형 (`Template-DoosanParallelPlace-v0`) | 튜닝 가이드라인 및 조정 사유 |
| :--- | :--- | :--- | :--- |
| **은닉층 용량 (`net`)** | `[256, 256, 128]` | `[512, 256, 128]` | Amodal 비전 마스크 및 렌치 계산 등으로 관측 차원이 대폭 늘어날 시 은닉층 용량을 확대하여 용량을 보정합니다. |
| **롤아웃 스텝 (`rollout_steps`)** | `24` | `16` | 고부하 비전/기하 데이터가 대규모 병렬 배치로 인입될 시 VRAM 포화를 막기 위해 스텝 크기를 하향 조정합니다. |
| **학습률 (`learning_rate`)** | `3.0e-4` | `2.5e-4` | 상태 차원 및 수식 복잡도가 높은 환경에서는 학습의 급격한 발산을 억제하기 위해 학습률을 보수적으로 낮춥니다. |

### **3. 자산 아카이빙 폴더 구조 관리 표준**
학습 실행 시 SKRL 내부 트레이너 엔진에 의해 `logs/skrl/franka_stack/[날짜_시간_ppo_torch]/checkpoints/` 디렉터리가 영구 빌드됩니다. 최고 누적 보상 도달 시 `best_agent.pt`가 자동 갱신되며, 2,000 스텝 주기마다 `agent_2000.pt`, `agent_4000.pt` 형식의 백업 파일이 영구 축적되어 실험 실패 시 즉각적인 롤백이 가능하도록 보장합니다.

---

## **4.3 실전 가동 및 자원 최적화 명령어 (Execution Blueprint)**

화면에 3D 그래픽을 렌더링하는 연산은 그래픽 카드의 셰이더 코어와 그래픽 메모리를 대량 잠식하는 주범입니다. 따라서 고속 본 학습 시에는 화면 출력을 차단하는 Headless 인자를 주입하고 병렬 환경 수를 극대화하는 전술을 취합니다.

### **1. 본 학습 가동 및 병렬 환경 수 자동 최적화 전술**
최초 가동 시 `--num_envs=16`으로 시작하여 하드웨어 크래시(OOM)가 발생하지 않는 선까지 32, 64, 128로 점진적으로 스케일을 확장하여 가동합니다. (※ 비전 연산이 혼재된 Doosan 심화 환경에서는 `--num_envs=32` 기동을 권장합니다.)

```bash
conda activate isaaclab
cd ~/franka_isaaclab

# Headless 모드로 GPU 자원을 극대화하여 PPO 학습 가동
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py \
  --task=Template-Stack-v0 \
  --num_envs=64 \
  --headless
```

### **2. 실시간 GPU 자원 점유율 및 CUDA 코어 연산 모니터링**
학습 스크립트를 실행한 직후 별도의 터미널 콘솔을 열어 실시간 자원 사용 현황을 확인합니다.
```bash
watch -n 0.5 nvidia-smi
```

### **3. 실험 자산 붕괴 시 특정 마일스톤 복구 (Resume) 명령어**
학습 진행 중 발산하거나 정책이 붕괴될 경우, `--resume` 인자와 백업 체크포인트를 결합하여 이전 모멘텀 상태를 복원하고 연속 학습을 수행합니다.
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py \
  --task=Template-Stack-v0 \
  --resume \
  --checkpoint=logs/skrl/franka_stack/[특정날짜폴더]/checkpoints/agent_20000.pt \
  --headless
```

---

## **4.4 시스템 검증 및 자가 진단 트러블슈팅 매트릭스**

End-to-End GPU 가속을 적극적으로 가동할 때 직면하게 되는 물리적 장치 한계와 연산 병목 현상을 해결하는 지침입니다.

| 장치/콘솔 크래시 현상 | 시스템 내부 원인 분석 (Root Cause) | 아키텍처 관점 즉각 조치 처방 (Action) |
| :--- | :--- | :--- |
| **Out of memory (OOM) 또는 Segmentation fault (core dumped) 발생 후 강제 종료** | 병렬 환경 개수(`num_envs`)를 하드웨어 물리 VRAM 용량 대비 과도하게 설정하여 메모리 주소 공간이 포화됨. | `--num_envs` 파라미터 수치를 절반으로 낮추고(예: 64 ➔ 32), 필요 시 YAML 내 `rollout_steps` 수치를 하향 튜닝합니다. |
| **`nvidia-smi` 모니터링 시 GPU 사용률(GPU-Util)이 10% 미만으로 정체되며 병목 발생** | 커스텀 보상/관측 연산 내부에 GPU 가속을 방해하는 파이썬 `for` 루프문이나 CPU 강제 전환 코드(`.cpu().numpy()`)가 삽입됨. | 수식 연산부 내 넘파이(NumPy) 및 CPU 연동부를 제거하고, 순수 PyTorch GPU 텐서 연산(`torch.where`, `torch.norm` 등)으로 재맵핑합니다. |
| **GUI 모드 실행 시 창이 뜨자마자 벌칸(Vulkan) 그래픽 드라이버 크래시 발생** | Xorg 세션 주사율 충돌 또는 웹 브라우저 등 기타 백그라운드 프로세스가 VRAM 자원을 사전 점유하여 충돌 발생. | 백그라운드 그래픽 점유 프로세스를 강제 종료하고, 시각적 디버깅 시에는 환경 개수를 `--num_envs=2` 수준으로 최소화하여 기동합니다. |

---

## **4.5 자가 검증 및 릴리즈 체크리스트**

고속화 및 자산 최적화 공정을 완수하고, Stage 1의 Franka 기반 RL 표준화 매뉴얼 라인업을 최종적으로 완성 및 릴리즈하기 위한 판정 기준입니다.

- [ ] **GPU 연산 효율성 검증**: Headless 모드 기반 본 훈련 실행 시, `nvidia-smi` 상에서 GPU 사용률(GPU-Util) 수치가 **80% ~ 95%** 영역을 안정적으로 채우며 연산 정체 현상 없이 연속 병렬 역전파를 수행하는가?
- [ ] **자산 아카이빙 안정성 검증**: `logs/` 디렉터리 내에 지정된 2,000 스텝 주기 분량의 디버깅 및 복구용 백업 모델 자산 파일들이 누락 없이 규칙적으로 정기 생성되는가?
- [ ] **릴리즈 모델 거동 평가**: 학습이 성공적으로 완료된 가중치 파일 `best_agent.pt`를 사용하여 `play.py` 스크립트를 구동했을 때, 로봇 암이 기계적 떨림 없이 매끄럽게(Smooth Motion) 목표 블록을 잡고 적재해 내는가?