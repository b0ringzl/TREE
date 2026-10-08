# D2g 论文证据整合与图表定稿阶段报告

完成时间：2026-08-23

## 阶段状态

D2g 已完成论文证据链整合、主图、统计主表、方法与结果初稿，并更新两周工作汇报 PPT。全程未启动训练、微调或修改模型环境。

## 证据链

- D2c：自然样本观察性匹配，用于发现现实场景现象及混杂边界。
- D2d：同树受控曝光，作为“严重曝光损害单影像、点云对曝光稳定、融合部分挽回”的主要因果证据。
- D2e：同树点云结构质量消融，作为“完整度与纯度是重要可控因素”的主要机制证据。
- D2f：241 株盲法人工复核，提供有限且受道路/类别混杂影响的真实样本外部支持。

## 结论口径

论文不再使用“任意过亮/过暗都会降精度”“点云质量只取决于分割精度”或“融合必然完全恢复”等绝对表述。正式口径限定为目标树区域的严重曝光信息损失、点云参数固定条件以及完整度/纯度的重要可控作用。

## 权威产物

- D2g 根目录：`lidar data/whu/derived/tscmdl/d2_exposure_stratified_evaluation/20260823_paper_evidence_package_v1`
- 主图：`figures/figure_d2g_evidence_chain.png`
- 证据链表：`tables/table_d2g_evidence_chain.csv`
- 主结果表：`tables/table_d2g_primary_results.csv`
- 方法与结果初稿：`论文方法与结果初稿.md`
- 图注与统计口径：`论文图注与统计口径.md`
- 机器验收：`validation.json`
- 生成脚本：`Lidar/tscmdl_whu_sandbox/scripts/build_d2g_paper_package.py`

## 后续建议

下一阶段应由论文写作需求决定：优先补充独立复核者一致性和真实自动分割结果，或直接进入论文图表编号、正文嵌入和补充材料编排。D2g 本身不需要继续训练。

