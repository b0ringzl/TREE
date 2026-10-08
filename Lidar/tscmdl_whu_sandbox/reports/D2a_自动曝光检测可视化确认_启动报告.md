# D2a 自动曝光检测可视化确认启动报告

启动时间：2026-08-19T11:15:03+08:00

## 一、阶段范围

本小阶段仅建立自动过暗/过亮结果的可视化确认入口，不启动训练，
不依据模型预测筛样，也不把用户登记的 7 棵论文案例作为实验划分。

## 二、输入与选择原则

- 来源清单：`C:\Users\25041725G\Desktop\receive\TREE\lidar data\whu\derived\tscmdl\d1_four_class_image_quality\candidate_manifest.json`
- 数据划分：仅 `test_silver`，未进入 D1 训练、验证或核心测试指标。
- 过暗：`dark_ratio > 0.35`。
- 过亮：`bright_ratio > 0.35`。

## 三、样本规模

- 过暗：37 张。
- 过亮：144 张。
- 合计：181 张。

## 四、产物

- 冻结协议：`C:\Users\25041725G\Desktop\receive\TREE\lidar data\whu\derived\tscmdl\d2_exposure_stratified_evaluation\20260819_visual_confirmation_v1\protocol.json`
- 可视化清单：`C:\Users\25041725G\Desktop\receive\TREE\lidar data\whu\derived\tscmdl\d2_exposure_stratified_evaluation\20260819_visual_confirmation_v1\visualization_manifest.json`
- 结构验证：`C:\Users\25041725G\Desktop\receive\TREE\lidar data\whu\derived\tscmdl\d2_exposure_stratified_evaluation\20260819_visual_confirmation_v1\validation.json`
- 用户确认记录由可视化服务首次启动时单独创建。

## 五、验证结果

- 样本键唯一。
- 所有样本均属于 test_silver。
- 所有图像与点云资产存在。
- 选择过程未读取模型预测。

## 六、风险与边界

自动曝光标签只描述像素亮暗比例，不等同于目标树可辨识性或配对质量。
本阶段的人工确认只验证曝光分组是否合理，不重新执行 D1 数据清洗。

## 七、下一阶段门

用户完成或抽查可视化确认后，另行批准冻结三模态推理清单；
在此之前不启动 D2 模型推理或训练。

## 八、可视化服务验证

- 服务地址：`http://127.0.0.1:8769/`
- 页面统计：181 张，其中过暗 37 张、过亮 144 张。
- API 分组计数与冻结清单一致。
- 首个过暗样本 `21_2_1596` 的原始裁剪与点云叠加均返回 HTTP 200。
- 浏览器级页面截图检查通过，筛选器、亮度指标、原图/点云切换和确认表单均正常显示。
- 用户确认记录当前为 0/181；验证过程未代替用户提交任何确认。
