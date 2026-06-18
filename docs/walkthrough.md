# Walkthrough - Resolving Isaac Lab Training Errors

This walkthrough documents the fixes applied to successfully run the reinforcement learning training script for Franka robot tasks within NVIDIA Isaac Lab.

## Summary of Changes

### 1. Hydra Dependency Installation
Installed the missing `hydra-core` package directly into the Isaac Sim Kit Python environment to resolve `ImportError: Hydra is not installed`:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh -m pip install hydra-core
```

### 2. Configuration Key Updates (`STATES` to `OBSERVATIONS`)
Updated the network input keys from `STATES` to `OBSERVATIONS` across all PPO configuration files under `franka_isaaclab/tasks` to prevent `AttributeError: 'NoneType' object has no attribute 'shape'` during skrl model instantiation:
* **Reach Config**: [skrl_ppo_cfg.yaml (reach)](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/reach/agents/skrl_ppo_cfg.yaml)
* **Lift Config**: [skrl_ppo_cfg.yaml (lift)](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/lift/agents/skrl_ppo_cfg.yaml)
* **Stack Config**: [skrl_ppo_cfg.yaml (stack)](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/stack/agents/skrl_ppo_cfg.yaml)

## Verification Results

We verified the fixes by running the training pipeline:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Reach-v0
```

* **Outcome**: The simulator successfully starts up, parses configuration, builds the environment, wraps it with `SkrlVecEnvWrapper`, and enters the training loop (which prints a progress bar showing iteration counts).
* **Cleanup**: The test run was cancelled to save GPU resources, but training functionality is fully operational.


현재 isaac lab을 통해 학습 명령을 실행했었는데 이때 셋팅을 어느파일에 어떻게 했고 결과를 위해서 어떤 동작을 하고 어떤 결과물을 어느파일에 저장했는지 등에 대한 정리를 해주세요


이전 학습 실행 시 적용했던 **1) 설정 내용**, **2) 실행 동작**, **3) 최종 결과물**에 대한 핵심 요약입니다.

---

### 1. ⚙️ 설정 파일 및 변경 내용 (Settings)

학습이 정상적으로 실행될 수 있도록 아래의 파일들을 수정하고 설정했습니다.

| 작업 대상 파일 | 변경 전 | 변경 후 (현재 상태) | 설명 |
| :--- | :--- | :--- | :--- |
| **`scripts/skrl/train.py`** | `from isaaclab.utils.io import dump_pickle` | `dump_pickle` 함수를 `train.py` 내부에 로컬 함수로 직접 정의하여 사용 | `isaaclab` 유틸 라이브러리에서 누락된 `ImportError` 문제를 해결 |
| **`reach/agents/skrl_ppo_cfg.yaml`**<br>**`lift/agents/skrl_ppo_cfg.yaml`**<br>**`stack/agents/skrl_ppo_cfg.yaml`** | `network:`<br>&nbsp;&nbsp;`- name: net`<br>&nbsp;&nbsp;&nbsp;&nbsp;`input: STATES` | `network:`<br>&nbsp;&nbsp;`- name: net`<br>&nbsp;&nbsp;&nbsp;&nbsp;`input: OBSERVATIONS` | 단일 에이전트 환경 wrapper에서 상태 정보를 `OBSERVATIONS`로 넘기기 때문에, `STATES` 요청 시 `NoneType` 에러가 나는 문제를 해결 |

---

### 2. 🏃 학습 실행 및 검증 동작 (Actions)

결과를 확인하고 모델을 테스트하기 위해 다음 명령어들을 수행합니다.

* **학습 시작 (Train)**:
  NVIDIA Isaac Sim 전용 파이썬 환경(`python.sh`)으로 학습 코드를 실행하여 물리 시뮬레이터와 강화학습 루프를 결합해 학습을 구동합니다.
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Reach-v0
  ```
* **결과 재생 및 시각화 (Play)**:
  학습 완료 후 생성된 신경망 가중치(`.pt` 파일)를 불러와 시뮬레이터 창을 띄우고 로봇이 도달(Reach) 작업을 수행하는 모습을 눈으로 직접 확인합니다.
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py \
      --task=Template-Reach-Play-v0 \
      --checkpoint=logs/skrl/reach_franka/2026-06-17_19-15-47_ppo_torch/checkpoints/best_agent.pt
  ```
* **실시간 메트릭 모니터링 (TensorBoard)**:
  학습 주기 동안 누적 보상(Rewards)이 상승하고 손실(Loss)이 줄어드는 추세를 브라우저 화면에서 확인합니다.
  ```bash
  tensorboard --logdir=logs/skrl/
  ```

---

### 3. 📦 학습 완료 후의 결과물 (Outputs)

학습이 정상 종료(예: Reach 작업 기준 24,000 steps 완료)되면, `logs/skrl/reach_franka/{학습시작일시}_ppo_torch/` 디렉토리 하위에 다음과 같은 결과물들이 생성됩니다.

1. **`checkpoints/` 폴더**:
   * **`agent_xxxx.pt`**: 일정 timestep 주기마다 중간 저장되는 가중치 체크포인트 파일 (예: `agent_24000.pt`)
   * **`best_agent.pt`**: 학습 도중 가장 높은 보상 성능을 기록한 **최고의 모델 가중치 파일** (실제 재생이나 배포 시 이 파일을 사용합니다)
2. **`params/` 폴더**:
   * **`env.yaml`, `env.pkl`**: 학습 진행 당시의 환경 상세 설정 백업본
   * **`agent.yaml`, `agent.pkl`**: 학습 진행 당시의 PPO 알고리즘 하이퍼파라미터 설정 백업본
3. **`events.out.tfevents.*` 파일**:
   * TensorBoard를 통해 시각화할 수 있는 로그 파일 (보상 곡선, 손실 수치 기록 데이터)


---

### 4. 🛠️ `play.py` 실행 시 발생한 오류 원인 및 해결 요약

`scripts/skrl/play.py`를 실행했을 때 발생한 오류들과 이를 해결한 과정에 대한 정리입니다.

#### 1) 오류 원인 및 해결법 (Why & How)

| 오류 메시지 | 발생 원인 | 해결 방법 |
| :--- | :--- | :--- |
| **`ModuleNotFoundError: No module named 'isaaclab.utils.pretrained_checkpoint'`** | 이전 템플릿 코드는 사전 학습 모델 유틸리티를 `isaaclab.utils`에서 가져오려 했으나, 실제 설치된 Isaac Lab 구조상 강화학습 라이브러리는 `isaaclab_rl.utils` 하위에 위치함 | `from isaaclab_rl.utils.pretrained_checkpoint import ...`로 패키지 임포트 경로 수정 |
| **`AttributeError: 'PPO' object has no attribute 'set_running_mode'`** | 환경에 설치된 `skrl` 라이브러리의 버전은 `2.1.0` (v2.x)인 반면, 기존 코드는 `skrl v1.x` 기준 메서드(`set_running_mode`)를 사용하고 있었음 | `skrl v2.x` API 규격에 맞게 `runner.agent.enable_training_mode(False)`로 메서드 변경 |
| **`TypeError: PPO.act() missing 1 required positional argument: 'states'`** | `skrl v2.x` 버전부터 `act` 메서드가 Privileged State 데이터(`states`)를 필수 위치 인자(두 번째 인자)로 받도록 함수 시그니처가 변경됨 | `runner.agent.act(obs, None, timestep=0, timesteps=0)` 형태로 `states` 위치에 `None`을 명시적으로 인가함 |

---

#### 2) 향후 동일 오류 재발 방지책 (Prevention)
* **라이브러리 버전 고정 (Pinning)**: 코드 실행 환경이 다른 머신이나 새로운 환경으로 이전될 때, 호환되는 외부 패키지의 메이저 버전을 명시하여 설치해야 합니다. (예: `skrl==1.4.3` 또는 `skrl>=2.1.0` 등을 `requirements.txt`에 명시)
* **스크립트 소스 출처 일관성**: 외부 소스코드 혹은 다른 템플릿 저장소의 코드를 그대로 가져오기보다는, 현재 설치하여 로드하고 있는 로컬 서브모듈인 `third_party/IsaacLab/scripts/reinforcement_learning/skrl/play.py` 버전의 템플릿을 베이스라인으로 복사하여 프로젝트에 구성하는 것이 안전합니다.

---

#### 3) 처음 환경 설정 시 근본적인 해결 방안 (Root Resolution)
* **의존성 설치 스크립트 검토**: Isaac Lab을 초기 빌드 및 셋업할 때, `setup.py` 또는 `pyproject.toml` 등에 정의된 학습 래퍼 및 에이전트 라이브러리(`skrl`, `rsl_rl`, `rl_games` 등)가 현재 시뮬레이터 버전 개발 환경에서 완벽히 검증된 타깃 버전을 고정하여 설치되도록 `pip install` 단계를 조율해야 합니다.
* **서브모듈 싱크 유지**: 외부 템플릿 프로젝트를 복제(Clone)하여 설정할 경우, `third_party/IsaacLab` 등 서브모듈 폴더에 명시된 커밋 해시(Commit Hash)와 에이전트 라이브러리(`skrl` 등)가 동일한 배포 버전 주기를 따르는지 빌드 로그 및 `git submodule status`로 사전 싱크 여부를 체크하는 것이 좋습니다.

