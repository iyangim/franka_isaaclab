# Franka Robot Manipulation with Isaac Lab: 실행 가이드 (Execution Guide)

이 문서는 **Franka Emika Panda** 로봇을 활용한 3가지 점진적 강화학습(Reinforcement Learning) 매니퓰레이션 태스크(Reach, Lift, Stack)를 NVIDIA Isaac Lab 환경에서 수행하기 위한 상세 실행 가이드입니다. 

본 프로젝트는 **SKRL** 라이브러리의 **PPO(Proximal Policy Optimization)** 알고리즘을 사용하며, 물리 엔진 연산 및 신경망 학습 전체를 GPU 가속으로 처리합니다.

이 가이드는 `smart-shelf-robot/third_party/IsaacLab` 폴더 내의 Isaac Lab 프레임워크와 `isaaclab` Conda 가상환경을 기반으로 작성되었습니다.

---

## 📌 목차
1. [프로젝트 구조 및 구성 요소](#1-프로젝트-구조-및-구성-요소)
2. [사전 준비 및 설치 (Setup)](#2-사전-준비-및-설치-setup)
3. [태스크 목록 및 Gym 환경 확인](#3-태스크-목록-및-gym-환경-확인)
4. [기본 환경 테스트 (더미 에이전트 실행)](#4-기본-환경-테스트-더미-에이전트-실행)
5. [Task 1, 2, 3 상세 실행 및 학습 가이드](#5-task-1-2-3-상세-실행-및-학습-가이드)
6. [TensorBoard 모니터링 및 결과 분석](#6-tensorboard-모니터링-및-결과-분석)
7. [데모 동영상 및 GIF 생성 가이드](#7-데모-동영상-및-gif-생성-가이드)
8. [커스텀 환경 추가 및 보상 함수 수정](#8-커스텀-환경-추가-및-보상-함수-수정)

---

## 1. 프로젝트 구조 및 구성 요소

이 프로젝트는 NVIDIA Isaac Lab의 매니저 기반 RL 환경(Manager-Based RL Environment) 방식을 따르고 있으며, 핵심 소스코드는 아래와 같이 모듈화되어 있습니다.

```
franka_isaaclab/
├── source/franka_isaaclab/franka_isaaclab/
│   ├── assets/robots/franka.py      # Franka 로봇 구성 설정
│   └── tasks/manager_based/
│       ├── reach/                   # Task 1: reach (도달)
│       │   ├── joint_pos_env_cfg.py # Joint Position 조인트 제어 설정
│       │   └── reach_env_cfg.py     # Reach MDP 설정 (관측, 보상 등)
│       ├── lift/                    # Task 2: lift (들어올리기)
│       │   ├── joint_pos_env_cfg.py # Lift Joint 제어 및 Dexterous Cube 설정
│       │   └── lift_env_cfg.py      # Lift MDP 설정
│       └── stack/                   # Task 3: stack (쌓기)
│           ├── stack_joint_pos_env_cfg.py # Stack 환경 초기화 및 두 개의 큐브 설정
│           └── stack_env_cfg.py     # 다단계 커리큘럼 보상 및 MDP 설정
├── scripts/
│   ├── list_envs.py                 # 등록된 환경 리스트업 스크립트
│   ├── random_agent.py              # 랜덤 액션 에이전트 테스트
│   ├── zero_agent.py                # 제로 액션 에이전트 테스트
│   └── skrl/
│       ├── train.py                 # SKRL 에이전트 학습 스크립트
│       └── play.py                  # 학습된 체크포인트 재생(추론) 스크립트
└── docs/
    └── DEMO_GUIDE.md                # 데모 GIF 생성 가이드
```

---

## 2. 사전 준비 및 설치 (Setup)

### Prerequisites (요구사항)
- **Ubuntu 20.04 이상** 혹은 호환되는 Linux 배포판
- **NVIDIA Isaac Lab** (`~/smart-shelf-robot/third_party/IsaacLab` 경로에 빌드된 환경 사용)
- NVIDIA 드라이버 및 CUDA 환경이 정상적으로 지원되는 **CUDA-capable GPU**
- **Conda** 가상환경 도구

### 가상환경 활성화 및 패키지 빌드
Isaac Lab 실행을 위해 conda 가상환경(`isaaclab`)을 활성화하고, 패키지를 에디터블 모드(-e)로 연동합니다. 

> [!IMPORTANT]
> 반드시 Isaac Sim 전용 `python.sh` 인터프리터를 사용해야 그래픽 드라이버 링크, CUDA 라이브러리 경로(`LD_LIBRARY_PATH`), Omniverse 코어 파이썬 모듈 경로(`PYTHONPATH`)가 올바르게 주입되어 `ModuleNotFoundError`를 예방할 수 있습니다.

```bash
# 1. conda 가상환경 활성화
conda activate isaaclab

# 2. franka_isaaclab 저장소 폴더로 이동
cd ~/franka_isaaclab

# 3. Isaac Sim 전용 python.sh 인터프리터를 활용하여 에디터블 모드로 패키지 설치
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh -m pip install -e source/franka_isaaclab
```

---

## 3. 태스크 목록 및 Gym 환경 확인

설치 및 연동이 완료되었으면 아래 명령어를 통해 프로젝트 내에 등록된 Gym 환경 리스트를 출력하여 연결성을 확인합니다.

```bash
# conda 가상환경이 활성화된 상태에서 실행
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/list_envs.py
```

성공적으로 실행되면, 아래와 같이 등록된 `Template-` 접두사를 가진 환경이 표 형태로 출력됩니다.

| 번호 | 환경 ID (Task Name) | 엔트리 포인트 (Entry Point) | 설정 파일 (Config Entry Point) |
|---|---|---|---|
| 1 | `Template-Reach-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `...reach.joint_pos_env_cfg:FrankaReachEnvCfg` |
| 2 | `Template-Reach-Play-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `...reach.joint_pos_env_cfg:FrankaReachEnvCfg_PLAY` |
| 3 | `Template-Lift-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `...lift.joint_pos_env_cfg:FrankaCubeLiftEnvCfg` |
| 4 | `Template-Lift-Play-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `...lift.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY` |
| 5 | `Template-Stack-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `...stack.stack_joint_pos_env_cfg:FrankaCubeStackEnvCfg` |
| 6 | `Template-Stack-Play-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `...stack.stack_joint_pos_env_cfg:FrankaCubeStackEnvCfg_PLAY` |

---

## 4. 기본 환경 테스트 (더미 에이전트 실행)

본격적인 학습 전에 시뮬레이터가 정상 구동되는지 확인하기 위해 **임의의 동작(Random Actions)**이나 **무동작(Zero Actions)**을 생성하는 에이전트를 가동해봅니다.

```bash
# 16개 환경에서 큐브 리프트 환경에 랜덤 액션 인가
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/random_agent.py --task=Template-Lift-v0 --num_envs=16

# 16개 환경에서 큐브 리프트 환경에 제로 액션 인가
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/zero_agent.py --task=Template-Lift-v0 --num_envs=16
```

---

## 5. Task 1, 2, 3 상세 실행 및 학습 가이드

각 태스크는 난이도가 순차적으로 상승하며, 관측 공간(Observation) 및 보상 구성(Reward shaping)이 다르게 구성되어 있습니다.

---

### 🟢 Task 1: Reach (목표 지점 도달)
* **목표**: Franka 로봇의 말단 장치(End-Effector, panda_hand)를 3차원 공간상의 무작위 타겟 위치로 이동시킵니다.
* **학습 스텝 수**: 약 24,000 timesteps 내외
* **주요 보상**: 말단 장치와 목표 지점 간의 L2 거리 감소에 비례하는 보상 (`tanh` 커널을 통한 세밀한 제어) 및 조인트 속도 페널티.

#### 1) 학습 실행 (Training)
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Reach-v0
```

#### 2) 결과 재생 및 추론 (Play)
학습된 최고의 가중치(`best_agent.pt`)를 가져와 50개의 재생용 병렬 환경에서 구동합니다.
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py --task=Template-Reach-Play-v0 \
    --checkpoint=logs/skrl/reach_franka/*/checkpoints/best_agent.pt
```

'''
Ran command: 

(doosan_env) iyangim@user-Vector-16-HX-AI-A2XWIG:~$ gcloud auth login && gcloud auth application-default login


Yes, I remember the commands we used in the previous session:

### 1. Training Commands
To start reinforcement learning training using the simulation-specific Python interpreter:
* **Reach Task**:
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Reach-v0
  ```
* **Lift Task**:
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Lift-v0
  ```
* **Stack Task**:
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Stack-v0
  ```

### 2. Play / Evaluation Commands
To run and visualize the trained agent from a checkpoint:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py --task=Template-Reach-Play-v0 --checkpoint=logs/skrl/reach_franka/*/checkpoints/best_agent.pt
```

### 3. Dependency Installation Command
To install dependencies directly to the Isaac Sim environment:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh -m pip install hydra-core
```

Which of these commands would you like to run or modify now?

'''
Ran command: `find . -maxdepth 4 -name "*.pt"`

Ran command: `find logs/ -name "*.pt"`

Ran command: `find logs/ -mmin -120`

Ran command: `ps aux | grep train.py`

Listed directory iyangim

Ran command: `tail -n 100 ~/.bash_history`

Ran command: `date`

Ran command: `find logs/ -type f -exec ls -lt {} + | head -n 30`

Ran command: `find . -not -path '*/.git/*' -type f -mmin -60`

Ran command: `find /tmp -name "*isaaclab*" -mmin -120 2>/dev/null`

Ran command: `ls -la /tmp/isaaclab/logs/ 2>/dev/null || echo "Not found"`

Viewed train.py:1-237

Viewed skrl_ppo_cfg.yaml:1-85


We have found the training results from your run in the workspace under `logs/skrl/reach_franka/`:

### 📊 Training Run: `2026-06-17_19-15-47_ppo_torch`
* **Status**: **Completed Successfully**
* **Total Timesteps**: **24,000** (as configured in `skrl_ppo_cfg.yaml`)
* **Saved Checkpoints**:
  * Final Checkpoint: `logs/skrl/reach_franka/2026-06-17_19-15-47_ppo_torch/checkpoints/agent_24000.pt`
  * Best Checkpoint: `logs/skrl/reach_franka/2026-06-17_19-15-47_ppo_torch/checkpoints/best_agent.pt`

---

### 1. How to Visualize the Trained Agent (Play)
You can play and visualize the behavior of the trained agent in the simulator using the following command:
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py \
    --task=Template-Reach-Play-v0 \
    --checkpoint=logs/skrl/reach_franka/2026-06-17_19-15-47_ppo_torch/checkpoints/best_agent.pt
```

### 2. How to Monitor Training Metrics (TensorBoard)
To view the training graphs (cumulative rewards, learning rate scheduler updates, policy loss, etc.):
```bash
tensorboard --logdir=logs/skrl/
```
Then open your browser and navigate to `http://localhost:6006`.

---

### 🔍 Did you run a different task?
If you ran the `train.py` command for **Lift** (`Template-Lift-v0`) or **Stack** (`Template-Stack-v0`) tasks just now, we do not see new log directories under `logs/skrl` for them yet. 

If the script failed, stopped early, or showed an error in your terminal, please paste the terminal output here so we can help troubleshoot!


---

### 🟡 Task 2: Lift (큐브 집어 올리기)
* **목표**: 테이블 위에 놓여 있는 DexCube를 그리퍼로 움켜쥐고 지정된 목표 높이(최소 0.04m 이상) 이상으로 들어올립니다.
* **학습 스텝 수**: 약 36,000 timesteps 내외
* **주요 보상**: 그리퍼와 큐브 거리 보상, 큐브 리프팅 달성 보상, 그리퍼 그립 안정성 보너스, 조인트 정밀도/부드러움 페널티.

#### 1) 학습 실행 (Training)
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Lift-v0
```

#### 2) 결과 재생 및 추론 (Play)
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py --task=Template-Lift-Play-v0 \
    --checkpoint=logs/skrl/franka_lift/*/checkpoints/best_agent.pt
```

---

### 🔴 Task 3: Stack (큐브 쌓기)
* **목표**: 테이블 위에 불규칙하게 놓인 두 개의 큐브(Cube 1: 파란색, Cube 2: 빨간색) 중 Cube 1을 집어 들어 Cube 2의 바로 위에 올려놓아 수직으로 정렬하여 탑을 쌓고 파지 상태를 풉니다.
* **학습 스텝 수**: 36,000+ timesteps (매니퓰레이션 중 가장 고난도)
* **특징**:
  * 다단계 학습 보상 설계(curriculum reward layout):
    * **Stage 1 (도달)**: Cube 1로 이동
    * **Stage 2 (그립/리프트)**: 그리퍼로 파지 후 상승
    * **Stage 3 (정렬)**: Cube 1을 Cube 2 방향으로 이동 및 정렬
    * **Stage 4 (적치)**: Cube 2 상단에 Cube 1 적치
    * **Stage 5 (분리)**: 큐브 적치 완료 후 파지 해제 및 복귀
  * 안정적인 쌓기 검증 기능이 포함되어 조인트 제어가 매우 정밀하게 수행되어야 합니다.

#### 1) 학습 실행 (Training)
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Stack-v0
```

#### 2) 결과 재생 및 추론 (Play)
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py --task=Template-Stack-Play-v0 \
    --checkpoint=logs/skrl/franka_stack/*/checkpoints/best_agent.pt
```

---

## 6. TensorBoard 모니터링 및 결과 분석

학습이 시작되면 실시간으로 누적 보상(Reward) 및 손실(Loss) 값 등의 메트릭이 기록됩니다. TensorBoard를 사용하여 웹 브라우저에서 학습 추세를 확인할 수 있습니다.

```bash
# TensorBoard 실행 (기본 포트: 6006)
tensorboard --logdir=logs/skrl/
```

- 웹 브라우저 주소창에 `http://localhost:6006` 을 입력하여 모니터링 페이지에 접속합니다.
- 각 Task별로 학습 과정에서 보상이 수렴해 나가는 모습을 비교 분석해 볼 수 있습니다.

---

## 7. 데모 동영상 및 GIF 생성 가이드

README 파일 또는 프레젠테이션용 데모 비디오(GIF)를 만드는 절차입니다. 

### 1단계: 플레이 스크립트를 통한 윈도우 생성
가장 최신 혹은 최고의 성능을 내는 `best_agent.pt` 파일을 파라미터로 넘겨 시뮬레이터를 띄웁니다.
```bash
~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/play.py --task=Template-Lift-Play-v0 \
    --checkpoint=logs/skrl/franka_lift/2025-12-07_19-34-59_ppo_torch/checkpoints/best_agent.pt
```

### 2단계: 화면 캡처 및 녹화 (OBS Studio 권장)
1. OBS Studio 설치:
   ```bash
   sudo add-apt-repository ppa:obsproject/obs-studio
   sudo apt update
   sudo apt install obs-studio
   ```
2. OBS 실행 후 소스(Sources)에 **윈도우 캡처(Window Capture)**를 추가하고 `Isaac Sim` 창을 지정합니다.
3. 녹화 형식(Recording Format)을 `MP4`로 설정하고 녹화를 시작합니다. 로봇이 성공적으로 수행하는 시점 전후로 **10초~30초** 가량 분량을 확보합니다.

### 3단계: MP4를 고화질 GIF로 변환
`ffmpeg`를 사용하여 고화질 팔레트를 입혀 압축된 GIF를 생성합니다.
```bash
# reach 태스크
ffmpeg -i reach_demo.mp4 -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 docs/media/franka_reach.gif

# lift 태스크
ffmpeg -i lift_demo.mp4 -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 docs/media/franka_lift.gif

# stack 태스크
ffmpeg -i stack_demo.mp4 -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 docs/media/franka_stack.gif
```

---

## 8. 커스텀 환경 추가 및 보상 함수 수정

더 고성능의 에이전트를 훈련시키기 위해 기존 환경을 커스텀하고자 하는 경우 아래 절차를 따릅니다.

1. **보상 함수 가중치 수정**: 
   - 각 태스크 폴더의 `mdp/rewards.py` 파일 또는 환경 구성 파일 (예: `stack_env_cfg.py` 의 `RewardsCfg` 클래스)에서 보상 항들의 가중치(`weight`) 및 조건 값들을 튜닝합니다.
2. **신규 에이전트 설정(Hyperparameters) 수정**:
   - `tasks/manager_based/<task_name>/agents/skrl_ppo_cfg.yaml` 에서 신경망 아키텍처, 학습률(Learning Rate), 에포크(Epoch) 수, 미니배치 사이즈 등을 조정할 수 있습니다.
3. **새로운 태스크 생성**:
   - `tasks/manager_based/` 아래에 신규 폴더를 생성하고, `__init__.py` 파일에 새로운 Gym 환경 ID를 등록합니다.
   ```python
   gym.register(
       id="Template-MyTask-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.mytask_env_cfg:FrankaMyTaskEnvCfg",
           "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
       },
   )
   ```



지금까지의 작업을 당신은 smart-shelf-robot 폴더내의 두산로봇에 적용했습니다. 
따라서 e0509로봇의 학습을 수행했습니다. 
이부분을 확인하여 맞다면 모든 작업파일을 smart-shelf-robot폴더에 위치하고 꼭 메인 폴더의 내용을 확인해주세요.

태스크 3 수행에 두산로봇에 적용하여 smart-shelf-robot폴더에서만 작업이 적용될 수 있도록 준비해주세요.
이번에도 꼭 작업을 수행하고 수동으로 진행할때, 동일하게 작업할 수 있는 작업명령들을 정리하고 왜 그렇게 수행했는지 정리해주세요.

그룹1의 태스크 reach, lift, stack 학습을 이전에 실행하는 것에 이어서 계속해주세요.
