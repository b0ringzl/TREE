"""Three-page Chinese report; no NC photos embedded or public image redistribution."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'
DEST = ROOT / 'output/pdf/珍稀树种新增样本精度验证_20261007.pdf'
GREEN = colors.HexColor('#173e34'); MUTED = colors.HexColor('#52675f')
LIGHT = colors.HexColor('#eef3ea'); LINE = colors.HexColor('#d3ded0')


def main():
    if DEST.exists(): raise FileExistsError('Never replace a delivered report silently')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads((DATA / 'RARE_RESULTS.json').read_text(encoding='utf-8'))
    pdfmetrics.registerFont(TTFont('CJK', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKbold', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
    styles = {
        'title': ParagraphStyle('title', fontName='CJKbold', fontSize=23, leading=34, textColor=GREEN, spaceAfter=14, wordWrap='CJK'),
        'h2': ParagraphStyle('h2', fontName='CJKbold', fontSize=13, leading=21, textColor=GREEN, spaceBefore=14, spaceAfter=8, wordWrap='CJK'),
        'body': ParagraphStyle('body', fontName='CJK', fontSize=10, leading=17, textColor=GREEN, spaceAfter=8, wordWrap='CJK'),
        'small': ParagraphStyle('small', fontName='CJK', fontSize=8.6, leading=14, textColor=MUTED, spaceAfter=7, wordWrap='CJK'),
        'cell': ParagraphStyle('cell', fontName='CJK', fontSize=9, leading=14, textColor=GREEN, wordWrap='CJK'),
    }
    story = []
    def p(text, style='body'):
        story.append(Paragraph(text, styles[style]))
    def table(values, widths):
        tab = Table([[Paragraph(str(x), styles['cell']) for x in row] for row in values], colWidths=widths, repeatRows=1)
        tab.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), LIGHT), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, -1), .4, LINE), ('TOPPADDING', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9), ('LEFTPADDING', (0, 0), (-1, -1), 8)]))
        story.append(tab); story.append(Spacer(1, 8))
    primary = data['primary']; aq = data['per_species']['Aquilaria sinensis']; ar = data['per_species']['Artocarpus hypargyreus']
    p('TREE SPECIES IDENTIFICATION  /  2026-10-07', 'small')
    p('珍稀树种新增样本<br/>精度验证汇报', 'title')
    p('土沉香有所提高，白桂木轻微退步；尚不能宣称稳定提升。', 'h2')
    p(f"本项目新增主测试 {primary['n']} 张自然植物照片，同图对比原始 BioCLIP 2.5 零样本分支与正式网页 SSTv1-H120。"
      f"合计识对 {primary['before_correct']} → {primary['after_correct']} 张：{primary['before_top1']:.2%} → {primary['after_top1']:.2%}，净变化 {primary['gain_percentage_points']:+.2f} 个百分点。")
    table([['树种 / N', 'BioCLIP 2.5', 'SSTv1-H120', '变化 / pp', '纠正 / 退步'],
        ['土沉香 / 154', '141 / 154<br/>91.56%', '147 / 154<br/>95.45%', '+3.90', '9 / 3'],
        ['白桂木 / 34', '33 / 34<br/>97.06%', '32 / 34<br/>94.12%', '-2.94', '0 / 1'],
        ['合并 / 188', '174 / 188<br/>92.55%', '179 / 188<br/>95.21%', '+2.66', '9 / 4']], [106, 94, 94, 83, 110])
    p(f"两树种等权平均：{data['macro_two_species_before']:.2%} → {data['macro_two_species_after']:.2%}，"
      f"仅 {data['macro_two_species_gain_percentage_points']:+.2f} 个百分点。土沉香样本较多，合并值不能代表白桂木也有进步。")
    p('为什么还不能称为稳定提升？', 'h2')
    p('188 张照片来自 106 条观察记录、67 名观察者。多张照片可能是同一棵树；按观察者整组进行 10,000 次配对 bootstrap，以减少重复照片带来的虚假确定性。')
    table([['指标', '变化的观察者整组 95% 区间 / pp'],
        ['土沉香', '-2.22 至 +8.88'], ['白桂木', '-6.67 至 +0.00'], ['合并主结果', '-2.52 至 +7.10']], [160, 327])
    p('合并区间跨 0。可以报告这批资料的实际识对数量增加，不能据此断言稳定、显著的泛化提升。', 'small')
    p('本次没有训练、筛除识别错误图片、修改置信度或更新 Demo。论文 4 个裁片另计，未混入上述结果。', 'small')
    story.append(PageBreak())

    p('02  /  数据与公平测试', 'small')
    p('先锁定数据，再比较模型', 'title')
    p('新增样本怎么采集与排除重复', 'h2')
    p('社区来源为 iNaturalist 官方 API，检索准确学名 Aquilaria sinensis 与 Artocarpus hypargyreus。要求至少两名当前独立物种鉴定者，至少一人不是上传者，无冲突鉴定；采集顺序在预测前固定。')
    p('比对本地历史训练、验证、来源及待审/拒绝记录中的观察 ID、图片 ID、SHA256 与 pHash（汉明距离 ≤4），排除已登记样本及近重复。最短边原生 ≥224 像素，排除极端曝光和非照片资料；保留有难度的远景、树皮及花果照片。')
    p('锁定主测试：土沉香 154 张 / 89 条观察 / 59 名观察者；白桂木 34 张 / 17 条观察 / 11 名观察者。另有 4 个合格论文照片裁片，来源仅 2 篇论文。', 'small')
    p('同样的输入、完整的竞争目录', 'h2')
    p('两分支使用相同冻结 BioCLIP 2.5 编码器、文本提示、温度及 146 个展示类别竞争，复用正式网页“原图中心 + 全图补边”各半概率路径。不是把候选范围缩小到两类。中心图单独诊断亦保留。')
    table([['仅中心图诊断', 'BioCLIP', 'SST', '变化 / pp'], ['土沉香', '94.16%', '94.81%', '+0.65'], ['白桂木', '97.06%', '94.12%', '-2.94']], [190, 100, 100, 97])
    p('原始比较对象是 BioCLIP 2.5 零样本分支，不是 BioCLIP 1；使用项目既有目录、提示与温度，不代表零样本配置的理论上限。第 1、中间、末尾样本额外调用正式 infer 路径核对概率分布。未引入 SAM、点云、跟踪或高分辨率方案。', 'small')
    p('香港与拍摄日期分层', 'h2')
    table([['子集', 'N', 'BioCLIP → SST'], ['香港来源', '161', '91.93% → 95.03%'], ['香港以外', '27', '96.30% → 96.30%'], ['拍摄日期 ≥2026-09-08', '0', '无可用样本，不估计']], [210, 60, 217])
    p('“新增”指对已登记的本地项目资料新增，不是刚拍摄。较早公开图片可能进入过基础模型预训练；本次无法证明 BioCLIP 从未见过。来源标签也不等于本项目专家核准真值。', 'small')
    story.append(PageBreak())

    p('03  /  论文补充、错误与可追溯性', 'small')
    p('论文照片未显示提升', 'title')
    p('原始图注确认树种、原生像素提取', 'h2')
    p('通过 PMC 官方公开云端数据集下载开放论文，核对 XML 图注与 CC 许可。PDF 内嵌原生照片按固定矩形无缩放提取；显微组织、图表、沉香产品及其他树种均排除。')
    table([['论文来源 / 图版', '照片数', 'BioCLIP → SST'],
        ['Genome sequence of the agarwood tree<br/>Fig.1(a)：成熟土沉香', '1', '1 / 1 → 1 / 1'],
        ['Comparative Analysis of Metabolome and Transcriptome<br/>Fig.1(A,D,E)：幼苗、果壳、种子', '3', '0 / 3 → 0 / 3'],
        ['论文补充合计（2 个来源）', '4', '1 / 4 → 1 / 4']], [315, 60, 112])
    p('白桂木属论文 Fig.1(I) 的自然照片原生分辨率不足，未放大后凑样本；奇楠幼苗论文为混合材料图版，未纳入。论文裁片数量不能当成独立树木数，4 张小测试也不能代表论文照片总体性能。', 'small')
    p('当前仍有错误，需要保留而非隐藏', 'h2')
    p('主测试 SST 仍错 9 张。土沉香 7 张主要混淆到 Syzygium、Ilex、Litchi、Cratoxylum；白桂木 2 张分别输出为 Machilus velutina 和 Morus alba。其中白桂木 1 张是由原来正确变为错误。所有错误与退步均在逐图结果中保留。')
    p('只含两个目标树种正样本，测得的是对应的 Top-1 召回率，无法估计在其他树种中的误报精确率。不外推到全部珍稀植物、146 类整体或连续街景，不将模型置信度当作准确率。', 'small')
    p('来源与授权', 'h2')
    p('1. 成熟土沉香：doi.org/10.1093/gigascience/giaa013（CC BY 4.0）<br/>'
      '2. 幼苗/果壳/种子：doi.org/10.3390/molecules29051075（CC BY 4.0）<br/>'
      '3. 白桂木属来源核查：doi.org/10.1093/sysbio/syaa073（CC BY-NC 4.0）<br/>'
      '4. 奇楠材料核查：doi.org/10.3390/plants14060896（CC BY 4.0）<br/>'
      '公开数据分发说明：pmc.ncbi.nlm.nih.gov/tools/pmcaws/', 'small')
    p('社区原图为 NC 许可，仅用于私人非商业科学验证，未进入训练或可分发 Demo 包；本 PDF 不嵌入 NC 照片。原图、水印、归属与来源链接保留于本机逐图页面。质量验收不是授权。', 'small')
    p('追溯目录：outputs/rare_species_fresh_validation_20261006_v1<br/>'
      '完整报告 REPORT_ZH.md；逐图页面 REPORT_ZH.html；概率 predictions.jsonl；审计 FINAL_AUDIT.json。<br/>'
      '物理模型备份：同目录 pre_training_backup；固定数据协议备份：evaluation_protocol_snapshot。正式部署文件未改变。', 'small')
    def footer(canvas, doc):
        canvas.setStrokeColor(LINE); canvas.line(54, 44, A4[0]-54, 44)
        canvas.setFont('CJK', 8); canvas.setFillColor(MUTED)
        canvas.drawString(54, 28, 'SSTv1-H120 · 来源标签验证 · 不代表全部珍稀树种准确率')
        canvas.drawRightString(A4[0]-54, 28, f'{doc.page}')
    doc = SimpleDocTemplate(str(DEST), pagesize=A4, leftMargin=54, rightMargin=54, topMargin=45, bottomMargin=62,
                            title='珍稀树种新增样本精度验证', author='Tree Species Identification')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('PDF CREATED', DEST)

if __name__ == '__main__': main()
