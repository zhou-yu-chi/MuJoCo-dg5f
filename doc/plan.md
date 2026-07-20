好，我們把模型訓練這件事完全獨立出來——先不碰真機、不碰 ROS2，只做「點雲進 → 抓取姿態出」這一件事。

我建議分成五個階段，每階段跑完都有可驗證的產出：

---

## 階段 0：環境準備

```bash
conda create -n dexgrasp python=3.10
conda activate dexgrasp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mujoco trimesh open3d numpy scipy tqdm
```

用 **MuJoCo** 而不是 Isaac Sim，理由：安裝快、接觸模擬夠準、單機就能跑、debug 容易。等資料生成流程確定了再考慮搬到 Isaac Lab 加速。

**產出**：`python -c "import mujoco; print(mujoco.__version__)"` 有輸出。

---

## 階段 1：把 DG-5F 載進 MuJoCo

URDF → MJCF 轉換，然後確認手指能動：

```python
# convert.py
import mujoco
m = mujoco.MjModel.from_xml_path("dg5f.urdf")   # MuJoCo 可直接讀 URDF
mujoco.mj_saveLastXML("dg5f.xml", m)
print("joints:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
```

接著要手動編輯 `dg5f.xml`：
- 加 `<option>` 設定：`cone="elliptic"`、`impratio="10"`（多指抓取的接觸穩定關鍵）
- 確認每個 collision mesh 都有（visual mesh 不參與接觸）
- 加一個 free joint 給手掌，讓整隻手可以在空間中自由擺放
- 確認 actuator 是 position control

**產出**：`python -m mujoco.viewer --mjcf=dg5f.xml`，能看到手、能拖動關節。這一步卡住的機率很高，卡住的話把 URDF 貼給我。

---

## 階段 2：物件資產

先只用 **5 個 YCB 物件**（不要一次全上）：

```
002_master_chef_can    圓柱
003_cracker_box        方盒
011_banana             不規則
025_mug                有把手
065-a_cups             小圓錐
```

從 YCB 官網下 `google_16k` 的 `.obj`，用 trimesh 做凸分解（MuJoCo 只支援凸體碰撞）：

```python
import trimesh
mesh = trimesh.load("mug.obj")
mesh.apply_scale(1.0)  # 確認單位是公尺
parts = trimesh.decomposition.convex_decomposition(mesh, maxhulls=32)
```

**產出**：5 個物件的 MJCF 片段，能跟手一起載入場景。

---

## 階段 3：抓取候選生成（核心）

這是整個流程最重要的一步。單次嘗試的邏輯：

```python
def try_grasp(model, data, obj_mesh):
    # 1. 在物件表面取樣一個點 p 與法向量 n
    p, face_idx = trimesh.sample.sample_surface(obj_mesh, 1)
    n = obj_mesh.face_normals[face_idx]

    # 2. 手掌沿 -n 方向退開 d (0.08~0.15m)，繞 n 隨機旋轉
    palm_pos = p - n * np.random.uniform(0.08, 0.15)
    palm_rot = random_rotation_about(n)

    # 3. 手指設為預張開姿態 (加隨機擾動)
    q_open = PREGRASP + np.random.normal(0, 0.1, n_joints)

    # 4. 設定狀態，沿 n 方向 approach 直到接觸
    #    再閉合手指 (position target 往閉合方向走)

    # 5. 擾動測試：物件重力 + 6 方向各推 1N，持續 1 秒
    #    物件位移 < 2cm → 成功

    return success, palm_pose, q_final
```

幾個實務要點：
- **成功率會很低**（純隨機約 2~10%），這正常。先用**反重力保持**再測擾動可以快一點
- 用 `mujoco.mj_step` 手動推進，不要用 viewer，速度差 100 倍
- 多進程跑，一個 CPU core 一個 MuJoCo instance

**產出**：跑滿 24 小時，收集到 **每個物件 2000+ 筆成功抓取**，存成：

```python
{
  "object_id": "025_mug",
  "object_pose": (4,4),        # 物件在世界座標
  "palm_pose": (4,4),          # 手掌 6D pose
  "joint_angles": (n_dof,),    # 最終關節角
  "contact_points": (k,3),     # 用來後續分析
}
```

---

## 階段 4：渲染點雲

用 MuJoCo 的 offscreen renderer，放一個虛擬相機在你 RealSense 的實際位置：

```python
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.enable_depth_rendering()
renderer.update_scene(data, camera="realsense")
depth = renderer.render()
```

然後：
1. 深度圖 → 點雲（用 RealSense D435 的真實內參：fx≈615, fy≈615, cx≈320, cy≈240）
2. 裁掉桌面平面
3. **加雜訊**：深度雜訊 σ=0.001×z²、隨機 dropout 5%、邊緣侵蝕
4. 降採樣到固定 2048 點

**產出**：每筆抓取對應一張 `(2048, 3)` 的點雲。

---

## 階段 5：訓練

```python
class GraspNet(nn.Module):
    def __init__(self, n_dof):
        self.backbone = PointNet2()          # 輸出 (B, 256, N_seed)
        self.score_head = MLP(256, 1)        # 可抓性
        self.rot_head   = MLP(256, 6)        # 6D rotation repr
        self.trans_head = MLP(256, 3)        # 相對 seed point 的偏移
        self.joint_head = MLP(256, n_dof)    # 關節角
```

Loss：
```
L = BCE(score, is_positive)
  + λ1 * geodesic_loss(R_pred, R_gt)      # 只算正樣本
  + λ2 * L1(t_pred, t_gt)
  + λ3 * L1(q_pred, q_gt)
```

負樣本從隨機取樣但驗證失敗的抓取來，正負比控制在 1:3。

**產出**：在 held-out 物件上，取 top-1 預測丟回 MuJoCo 驗證，成功率 > 40% 就算跑通了。

---

## 現在該做的

從階段 1 開始。請你先做兩件事：

1. 把 DG-5F 的 URDF 路徑結構貼給我（`tree` 一下，我要看 mesh 檔在哪、有沒有 collision mesh）
2. 跑一次 `mujoco.MjModel.from_xml_path("你的.urdf")`，把錯誤訊息或成功的 joint list 貼上來

我根據結果幫你把 MJCF 調好，然後我們往階段 2 走。