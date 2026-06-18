# 문제 해결 보고서: Isaac Lab 환경 및 학습 오류 해결

## 1. 요약 (Executive Summary)

본 보고서는 NVIDIA Isaac Lab 환경에서 Franka 로봇 작업의 강화학습 학습 스크립트 실행 시 발생했던 두 가지 연속적인 오류의 해결 과정을 문서화한 것입니다.

1. **`ImportError: Hydra is not installed`** (Hydra 패키지 누락)
2. **`AttributeError: 'NoneType' object has no attribute 'shape'`** (네트워크 초기화 단계에서의 입력 형태 오류)

현재 두 가지 문제 모두 해결되었습니다. 학습 파이프라인이 정상적으로 시작되고 시뮬레이션 환경이 구축되며, 정책(Policy) 및 가치(Value) 네트워크가 올바르게 초기화되어 학습 반복(Iteration)이 정상 작동하는 것을 확인했습니다.

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

---

## 3. 단계별 해결 방법

### 1단계: `hydra-core` 설치
시뮬레이터 내 파이썬 환경에 패키지를 설치하였습니다:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh -m pip install hydra-core
```

### 2단계: 설정 파일 업데이트
`franka_isaaclab/tasks` 경로 하위에 있는 3개의 PPO 설정 파일 내에서 정책(Policy) 및 가치(Value) 네트워크의 `input: STATES` 설정을 `input: OBSERVATIONS`로 변경하였습니다.

1. **Reach 작업 설정 (Reach Task Config):**
   * 파일 위치: [/home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/reach/agents/skrl_ppo_cfg.yaml](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/reach/agents/skrl_ppo_cfg.yaml)
   * 수정 사항:
     ```diff
          network:
            - name: net
     -        input: STATES
     +        input: OBSERVATIONS
              layers: [64, 64]
     ...
          network:
            - name: net
     -        input: STATES
     +        input: OBSERVATIONS
              layers: [64, 64]
     ```

2. **Lift 작업 설정 (Lift Task Config):**
   * 파일 위치: [/home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/lift/agents/skrl_ppo_cfg.yaml](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/lift/agents/skrl_ppo_cfg.yaml)
   * 수정 사항: `policy` 및 `value` 네트워크 하위의 `input` 값을 `STATES`에서 `OBSERVATIONS`로 변경.

3. **Stack 작업 설정 (Stack Task Config):**
   * 파일 위치: [/home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/stack/agents/skrl_ppo_cfg.yaml](file:///home/iyangim/franka_isaaclab/source/franka_isaaclab/franka_isaaclab/tasks/manager_based/stack/agents/skrl_ppo_cfg.yaml)
   * 수정 사항: `policy` 및 `value` 네트워크 하위의 `input` 값을 `STATES`에서 `OBSERVATIONS`로 변경.

---

## 4. 검증 결과 (Verification)

변경을 적용한 후 학습 명령어를 재실행하여 검증을 진행했습니다:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Reach-v0
```

### 검증 로그 (일부 발췌)
```
[INFO]: Completed setting up the environment...
[skrl:INFO] Environment wrapper: Isaac Lab (single-agent)
[skrl:INFO] Seed: 42
[INFO] Logging experiment in directory: /home/iyangim/franka_isaaclab/logs/skrl/reach_franka
...
  0%|▏                                     | 186/24000 [00:14<25:25, 15.61it/s]
```

오류 없이 시뮬레이션 환경 빌드가 완료되었으며, 초당 약 **15 Iterations (it/s)**의 속도로 학습 연산 루프가 정상적으로 수행되는 것을 확인했습니다.
