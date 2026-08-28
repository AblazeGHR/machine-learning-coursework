# MNIST TCAV 总报告（中文）

## 1. 报告范围

本总报告整合以下两组实验输出：

- 单概念基线：outputs/run_tcav/report/tcav_report.md 与 summary.json
- 多概念（PyTorch CAV）：outputs/stroke_concepts_pytorch/report/stroke_concepts_report_pytorch.md 与 stroke_concepts_summary_pytorch.json

## 2. 总体结论（先看）

- 两组实验共享同一分类模型精度：test accuracy = 0.9876，说明分类器本身性能稳定。
- X-shape 在单概念与多概念实验中都没有表现为“正向依赖概念”。
- 在更高 runs/样本量（PyTorch CAV, runs=20, concept_size=400, eval_size=512）下，结果显示：
  - x_shape 与 loop 对数字 8 均为显著负向依赖（mean < 0.5 且 p(mean < 0.5) < 0.05）。
  - 其余直线相关概念（slash/vertical/horizontal/backslash）对数字 8 不显著。
- 数字 3 在多个直线概念上表现出更高 TCAV（尤其 horizontal、backslash、vertical），提示当前概念构造下模型更倾向把这些方向性笔画与 3 的判别相关联。

## 3. 关键结果汇总

### 3.1 单概念 X-shape（run_tcav）

- score8_mean = 0.0391，95% CI = [-0.0183, 0.0965]
- score3_mean = 0.0187，95% CI = [-0.0172, 0.0547]
- gap(8-3) = 0.0203

解读：

- 两个类别的均值都远小于 0.5，且置信区间覆盖 0，说明“X-shape 促进该类别 logit”的证据非常弱。
- 尽管 8 比 3 略高，但差值很小，不支持“8 强依赖 X-shape”的结论。

### 3.2 多概念（PyTorch CAV，runs=20）

按 score8_mean 排序（报告原始排序）：

- slash: mean8=0.5017（不显著）
- vertical: mean8=0.4895（不显著）
- horizontal: mean8=0.3729（不显著）
- backslash: mean8=0.3375（不显著）
- x_shape: mean8=0.1451（负向显著，p_lt≈2.88e-7）
- loop: mean8=0.1080（负向显著，p_lt≈2.10e-14）

解读：

- 更高采样规模下，仅 x_shape 与 loop 呈现显著负向依赖。

## 4. 单概念基线与 PyTorch CAV 的互补结论

一致点：

- x_shape 对数字 8 不是正向关键概念。
- 在多概念 PyTorch CAV 中，x_shape 与 loop 对数字 8 均表现为负向显著。

互补信息：

- 单概念基线给出 X-shape 在目标类别上的整体效应强度（均值、置信区间与 8-3 差值）。
- PyTorch CAV 多概念实验给出跨概念排序与显著性，显示多个直线概念对数字 8 并不显著，而数字 3 上可能更高。

## 5. 统计与方法层面的解释

- 当前显著性规则是单侧检验相对 0.5 阈值：
  - 正向依赖：mean > 0.5 且 p(mean > 0.5) < 0.05
  - 负向依赖：mean < 0.5 且 p(mean < 0.5) < 0.05
- 当 runs、concept_size、eval_size 同时增大时，结论通常更稳健，本项目中 PyTorch CAV 结果可作为多概念主结论依据。

## 6. 实验结论（可直接引用）

- 在本实验设置下，数字 8 的判别并不依赖“X 形状”概念；相反，X-shape 更接近负向概念。
- 与其关注“X 形状”，当前模型行为更像依赖其他与 8 结构更相关的局部形态（例如环形闭合、上下半环连接等；现有概念集尚未直接覆盖）。
- 因此，当前概念库对“解释数字 8 的决策机制”覆盖不充分，建议扩展概念定义后再做 TCAV 验证。

## 7. 下一步建议

- 扩展概念集：补充“上环/下环”“双环闭合”“中部收腰”“交叉节点”等更贴近 8 的形态概念。
- 使用真实样本构造概念集：从数据集中人工筛选代表性笔画片段，而非仅规则生成。
- 在多瓶颈层重复 TCAV：比较不同层的概念可解释性差异。
- 固定随机种子并多次重复完整实验：报告均值与置信区间，降低偶然性。
- 增加对照分析：可加入 digit 0、6、9，观察闭环类数字与 8 的概念区分边界。
