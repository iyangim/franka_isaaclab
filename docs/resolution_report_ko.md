# 문제 해결 보고서: Isaac Lab 환경 및 학습 오류 해결

## 1. 요약 (Executive Summary)

본 보고서는 NVIDIA Isaac Lab 환경에서 Franka 로봇 작업의 강화학습 학습(train.py) 및 재생(play.py) 스크립트 실행 시 발생했던 오류들의 원인과 해결 과정을 문서화한 것입니다.

1. **`ImportError: Hydra is not installed`** (Hydra 패키지 누락)
2. **`AttributeError: 'NoneType' object has no attribute 'shape'`** (네트워크 초기화 단계에서의 입력 형태 오류)
3. **`play.py` 실행 시의 의존성 패키지 경로 및 `skrl v2.x` API 호환성 오류**

현재 위의 문제들이 모두 해결되어 학습 파이프라인의 구동뿐만 아니라 학습 완료 모델 가중치(`best_agent.pt`)를 시뮬레이터 상에서 시각화하여 확인하는 평가(Play) 기능까지 정상적으로 작동함을 검증하였습니다.

---

## 2. 근본 원인 분석 (Root Cause Analysis)

### 문제 1: Hydra 라이브러리 누락 (`ImportError`)
* **현상:** `scripts/skrl/train.py` 실행 시 python이 `hydra` 모듈을 임포트하지 못하고 에러를 출력했습니다.
* **원인:** Isaac Lab은 자체 빌드된 내장 Python 가상 환경(시뮬레이터 내부 폴더에 위치한 `_isaac_sim/python.sh`를 통해 실행됨)을 사용합니다. `isaaclab_rl` 패키지의 의존성 목록에 `hydra-core`가 정의되어 있었지만, 시뮬레이터에서 실행하는 특정 가상 환경에는 실제 설치가 이루어지지 않은 상태였습니다.
* **해결법:** `python.sh`가 가리키는 시뮬레이터 파이썬 가상 환경에 `hydra-core`를 직접 설치하였습니다.

### 문제 2: 설정 파일 키 이름 불일치 (`skrl` 내 `AttributeError`)
* **현상:** Hydra 에러 해결 후 스크립트를 재실행했을 때, 네트워크 초기화 단계에서 아래와 같은 오류가 발생하며 즉시 종료되었습니다.
  ```python
  File "/home/iyangim/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/torch/nn/modules/linear.py", line 286, in initialize_parameters
    self.in_features = input.shape[-1]
  AttributeError: 'NoneType' object has no attribute 'shape'
  ```
* **원인:** 
  * 커스텀 워크스페이스(`franka_isaaclab`) 내의 학습 설정 파일에 정책 및 가치 네트워크의 입력값으로 `input: STATES`가 지정되어 있었습니다.
  * 커스텀 워크스페이스에서는 시뮬레이션 환경을 `SkrlVecEnvWrapper`로 래핑하여 사용합니다. 최신 `skrl` 라이브러리(v2.x) 래퍼 구조에서 일반 환경의 관측값(Observation)은 `"observations"`라는 키로 맵핑됩니다.
  * 별도의 비대칭 상태(privileged/asymmetric state) 공간이 정의되지 않았기 때문에, `STATES` 키를 요청하면 `skrl`은 `None` 데이터를 넘겨주게 됩니다. 이에 따라 PyTorch가 첫 번째 `LazyLinear` 레이어를 생성할 때 입력 텐서가 `None`으로 전달되면서 `AttributeError`가 발생한 것입니다.
* **해결법:** 워크스페이스의 YAML 설정 파일 내 모든 네트워크 입력값 설정을 `STATES`에서 Isaac Lab의 표준 설정인 `OBSERVATIONS`로 변경했습니다.

### 문제 3: 재생 스크립트 실행 시 의존성 패키지 경로 및 `skrl v2.x` API 호환성 오류
* **현상:** 학습 결과를 재생(play.py)하는 과정에서 연속적으로 임포트 에러, 속성 에러, 매개변수 에러가 차례로 보고되었습니다.
  * `ModuleNotFoundError: No module named 'isaaclab.utils.pretrained_checkpoint'`
  * `AttributeError: 'PPO' object has no attribute 'set_running_mode'`
  * `TypeError: PPO.act() missing 1 required positional argument: 'states'`
* **원인:**
  * 기존 코드 템플릿은 사전 학습 유틸리티 모듈의 임포트 경로를 `isaaclab.utils`로 참조하였으나, 실제 모듈은 `isaaclab_rl.utils` 하위에 내장되어 있었습니다.
  * 실행 환경의 `skrl` 메이저 버전이 `2.1.0` (v2.x)인 반면, 템플릿의 API 코드는 `skrl v1.x` 기준(`set_running_mode` 사용, `act` 함수 내 `states` 파라미터 미인가)으로 고착화되어 발생한 호환성 불일치 문제였습니다.
* **해결법:** 임포트 구문을 변경하고, `enable_training_mode(False)`를 호출하며, `act` 함수에 `None`으로 `states` 위치 인자를 명시 지정하여 문제를 해결했습니다.

---

## 3. 단계별 해결 방법

### 1단계: `hydra-core` 설치
시뮬레이터 내 파이썬 환경에 패키지를 설치하였습니다:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh -m pip install hydra-core
```

### 2단계: 학습 설정 파일 업데이트
`franka_isaaclab/tasks` 경로 하위에 있는 3개의 PPO 설정 파일 내에서 정책(Policy) 및 가치(Value) 네트워크의 `input: STATES` 설정을 `input: OBSERVATIONS`로 변경하였습니다.

* 대상 파일 및 수정 내역:
  * [Reach 설정 파일](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/reach/agents/skrl_ppo_cfg.yaml)
  * [Lift 설정 파일](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/lift/agents/skrl_ppo_cfg.yaml)
  * [Stack 설정 파일](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/stack/agents/skrl_ppo_cfg.yaml)

```diff
     network:
       - name: net
-        input: STATES
+        input: OBSERVATIONS
         layers: [64, 64]
```

### 3단계: 재생 스크립트 수정 (`scripts/skrl/play.py`)
`skrl v2.x` API 및 로컬 모듈 구조와 매치되도록 아래와 같이 수정사항을 반영하였습니다:

```diff
# 1. 임포트 경로 수정
-from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
+from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# 2. 에이전트 평가 모드 토글 함수 수정
-runner.agent.set_running_mode("eval")
+runner.agent.enable_training_mode(False)

# 3. act 함수 호출 시 states 위치 인자에 None 명시 지정
-outputs = runner.agent.act(obs, timestep=0, timesteps=0)
+outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
```

---

## 4. 검증 결과 (Verification)

### 1) 학습 단계 (Train)
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Reach-v0
```
* **결과:** 오류 없이 환경 빌드가 완료되었으며, 초당 약 **15 it/s** 속도로 정상 학습이 진행됨을 확인했습니다.

### 2) 시각화 재생 단계 (Play)
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py \
    --task=Template-Reach-Play-v0 \
    --checkpoint=logs/skrl/reach_franka/2026-06-17_19-15-47_ppo_torch/checkpoints/best_agent.pt
```
* **결과:** 정상적으로 시뮬레이터 창이 실행되며, PPO 신경망 모델 가중치를 안정적으로 로딩하고 물리 시뮬레이션 인터페이스와 동기화되어 학습 결과를 시각적으로 성공적으로 재생하였습니다.
