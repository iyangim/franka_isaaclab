# **[제 1장] Isaac Lab 기반 프란카 로봇 기초 셋업 및 물리 무결성 검증**
## *(Phase 1: Environment Setup & Sanity Check)*

NVIDIA Isaac Lab 환경을 활용하여 편의점 매대 진열 로봇 AI 학습 공간을 독자적으로 구성하고, 본격적인 강화학습(RL) 학습을 시작하기 전 시뮬레이터 물리 엔진의 정역학/동역학 신뢰성을 검증하는 단계입니다. 본 장은 에디터블 패키지 설치부터 가상환경 연동, GUI 디버깅 모션 평가 및 초기 보상 베이스라인 검측까지의 필수적인 표준 운영 절차(SOP)를 제공합니다.

---

## **1.1 python.sh 인터프리터 구동의 당위성 (이론적 바탕)**

NVIDIA Isaac Sim은 단순한 파이썬 라이브러리가 아니라, **Pixar USD(Universal Scene Description) 기반의 거대한 독립형 시뮬레이션 플랫폼**입니다. 일반 시스템 python이나 일반 Conda 환경의 인터프리터로 실행하면 Omniverse 및 Isaac Sim의 핵심 C++ 바이너리와 라이브러리(`omni`, `pxr` 등)를 로드하지 못해 즉각 `ModuleNotFoundError` 또는 `Segmentation Fault` 크래시를 뱉게 됩니다.

* **가상 경로 구성**: 일반 `python` 명령어는 시스템 환경변수만 참조하지만, Isaac Sim 폴더 내부에 포함된 `_isaac_sim/python.sh`는 실행되는 순간 **컨테이너화된 가상 경로를 구성**합니다.
* **래핑(Wrapping) 가교 역할**: 자사 전용으로 빌드된 고성능 그래픽 드라이버 링크, 하드웨어 가속용 CUDA 라이브러리 경로(`LD_LIBRARY_PATH`), 그리고 Omniverse 코어 파이썬 모듈 경로(`PYTHONPATH`)를 런타임에 자동으로 주입해 줍니다.
* **표준 연동 절차**: 따라서 본 교재의 모든 실습은 프로젝트 내장 가상 환경 경로인 `~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh`를 명시하여 기동하는 것을 절대적인 표준으로 선언합니다.

---

## **1.2 독립된 실습 공간 구축 및 환경 연결 (Setup)**

**[실행 의미]**
기존 `smart-shelf-robot` 프로젝트 내부에 빌드된 Isaac Lab 코어를 빌려와 쓰되, 개발 중인 `franka_isaaclab` 소스 코드를 에디터블(`-e`) 모드로 설치하여 별도의 빌드 과정 없이 코드가 시뮬레이터에 즉각 반영되도록 레지스트리에 등록합니다.

**[실행 명령어]**

### **1. 프로젝트 복사 및 작업 공간 이동**
```bash
# 홈 디렉터리로 이동 및 리포지토리 클론
cd ~
git clone https://github.com/dtruongthinh2409/franka_isaaclab.git

# 실습용 프로젝트 폴더 진입
cd ~/franka_isaaclab
```

### **2. 패키지 레지스트리 등록**
사용자의 환경 스타일에 따라 **방법 A** 또는 **방법 B** 중 하나를 선택하여 실행합니다.

* **방법 A: Conda 가상환경(`isaaclab`)을 사용하는 경우 (권장)**
  가상환경을 활성화한 상태에서 전용 `python.sh` 인터프리터로 에디터블 모드 패키지를 설치합니다.
  ```bash
  conda activate isaaclab
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh -m pip install -e source/franka_isaaclab
  ```

* **방법 B: Isaac Lab 쉘 래퍼(`isaaclab.sh`)를 사용하는 경우**
  기존 `smart-shelf-robot` 내부에 있는 `isaaclab.sh` 파일 경로를 직접 지정하여 패키지를 등록합니다.
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/isaaclab.sh -p -m pip install -e source/franka_isaaclab
  ```

### **3. 설치 검증**
새로운 프로젝트 공간에서 기존 Isaac Lab이 연결되어 태스크들을 정상적으로 인식하는지 확인합니다.

* **Conda 환경 기반:**
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/list_envs.py
  ```
* **쉘 래퍼 기반:**
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/isaaclab.sh -p scripts/list_envs.py
  ```

* **성공 판정 기준 (Success Metric)**: 검증 명령어를 실행했을 때, 터미널 로그 최하단 Gymnasium 등록 환경 목록에 `Template-Reach-v0`, `Template-Lift-v0`, `Template-Stack-v0`가 에러 없이 표시되어야 합니다.

---

## **1.3 무작위 요원(Random Agent)을 통한 물리 엔진 시각 검증**

**[실행 의미]**
강화학습 뇌(Policy)를 개입시키지 않고, 타임스텝마다 완전히 무작위 액션(Random Action)만을 주어 로봇 강체와 오브젝트 간의 초기 레이아웃, 질량, 마찰력 등 물리 역학이 시뮬레이션 세계관 안에서 무너지지 않는지 GUI로 계측합니다.

**[실행 명령어]**

* **Conda 환경 기반:**
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/random_agent.py --task=Template-Reach-v0 --num_envs=4
  ```
* **쉘 래퍼 기반:**
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/isaaclab.sh -p scripts/random_agent.py --task=Template-Reach-v0 --num_envs=4
  ```
> 💡 **VRAM 절약 팁**: 리포지토리 기본값대로 수백~수천 개의 병렬 환경을 한 번에 띄우면 그래픽 메모리(VRAM) 부족으로 화면이 즉시 튕기거나 멈출 수 있습니다. 직접 눈으로 모니터링하며 디버깅할 때는 환경 개수 인자를 `--num_envs=4`와 같이 4개~16개 사이로 제약하여 실행하는 것이 필수입니다.

* **필수 시각 검증 체크리스트**:
  - [ ] **Franka 로봇 고정 상태**: Franka 로봇 베이스가 테이블 위에 뜨거나 파묻히지 않고 단단히 고정되어 기단 결합부가 고정 상태를 유지하는가?
  - [ ] **목적물 가동 영역**: 목표 구체(Target)가 로봇이 팔을 뻗어 상호작용할 수 있는 가동 범위(Reachable Area) 내에 정상적으로 안착 및 스폰되는가?
  - [ ] **강체 충돌 반응**: 무작위 기동 시 로봇 손가락이 구체를 타격할 때, 구체가 시뮬레이터 바닥을 관통하거나 우주로 튕겨 나가지 않고 마찰 마운트 법칙에 따라 자연스럽게 밀려 나가는가?

---

## **1.4 정지 요원(Zero Agent)을 통한 베이스라인 보상 검증**

**[실행 의미]**
로봇에게 제로 액션(Zero Action, `[0, 0, ...]`)을 주어 완전히 고정된 준비 자세를 유지시킵니다. 가만히 멈춰 있을 때의 기초 보상 흐름을 관찰하여, 환경 초기화 시점에 하드웨어 보호용 액션 패널티가 과도하게 작용하여 시작부터 마이너스 보상이 무한 누적되는 보상 설계적 불량이 존재하지 않는지 파악하는 베이스라인 측정 단계입니다.

**[실행 명령어]**

* **Conda 환경 기반:**
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/zero_agent.py --task=Template-Lift-v0
  ```
* **쉘 래퍼 기반:**
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Template-Lift-v0
  ```

* **터미널 로그 검증 체크리스트**:
  - [ ] **누적 보상 평탄화**: 로봇이 완전히 멈춘 정지 상태에서 터미널 실시간 리워드(Reward) 출력값이 `0` 근처에서 큰 요동 없이 안정적으로 유지되는가?
  - [ ] **초기 패널티 상한**: 시작 자세에서 가해지는 조인트 보호 패널티 과다로 인해 `-50` 이상의 큰 음수 누적 리워드가 계속 쌓이고 있지는 않은가?

---

## **1.5 GUI 디버깅 화면 제어 및 트러블슈팅 SOP**

시뮬레이터 창이 열리는 GUI 환경에서 튕김 현상 또는 무반응 오류가 발생할 때 하드웨어 레벨에서 대처하는 가이드라인입니다.

### **1.5.1 --headless 옵션의 최적 운용**
* **초기 디버깅 (화면 활성화)**: 알고리즘이 물리 엔진과 올바르게 반응하는지 첫 1~2분간 모니터링할 때는 `--num_envs=4` 인자로 화면을 직접 띄워 감상합니다.
* **고속 본 학습 (화면 비활성화)**: 본격적인 36,000 스텝 이상의 수렴 훈련을 실행할 때는 3D 렌더링에 의한 GPU 자원 및 메모리 잠식을 전면 차단하기 위해 무조건 `--headless` 옵션을 지정하여 훈련 속도를 수십 배 가속시킵니다.
  ```bash
  ~/smart-shelf-robot/third_party/IsaacLab/_isaac_sim/python.sh scripts/skrl/train.py --task=Template-Lift-v0 --headless
  ```

### **1.5.2 Vulkan 그래픽스 드라이버 크래시 및 튕김 대응**
* **원인**: 백그라운드 프로세스가 VRAM을 선점하고 있거나 병렬 환경 개수가 대역폭을 초과했을 때 벌칸(Vulkan) 드라이버 파열로 인해 `Segmentation fault (core dumped)` 에러가 나며 창이 튕깁니다.
* **조치**: 크롬 웹 브라우저나 타 가중치 훈련 모듈 등 VRAM을 소모하는 응용 프로그램을 완전히 강제 종료한 뒤, `--num_envs` 파라미터 수치를 1 또는 2로 더 줄여 재기동합니다.

### **1.5.3 원격 서버 환경(SSH)에서의 디스플레이 오류**
* **원인**: 호스트 PC(Ubuntu 데스크톱)가 아닌 SSH 원격 접속 콘솔에서 GUI 환경을 트리거하면 디스플레이 장치가 매핑되지 않아 X11 관련 에러가 발생합니다.
* **조치**: 반드시 모니터가 연결된 우분투 내부 터미널 환경에서 실행하거나, 원격 환경인 경우 가상 디스플레이(VNC 또는 Xorg 세션)가 완비되지 않았다면 무조건 `--headless` 옵션을 인자로 주어 실행해야 런타임 크래시를 방지할 수 있습니다.