"""Render an evidence-led Chinese inventory report from read-only audit records."""
import json,html
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/hk_data_inventory_20260912_v1'
PDF=ROOT/'output/pdf/Hong_Kong_Data_Inventory_20260912.pdf'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
inv=read(OUT/'inventory.json');audit=read(OUT/'content_audit_v2.json');stats=inv['inventories']
def gb(v):return f'{v/1e9:,.2f}'
def num(v):return f'{v:,}'
def source(suffix):return next(v['data'] for k,v in inv['important_sources'].items() if k.replace('\\','/').endswith(suffix))
pdfmetrics.registerFont(TTFont('Chinese','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('ChineseBold','C:/Windows/Fonts/msyhbd.ttc'))
green=colors.HexColor('#245B48');muted=colors.HexColor('#64756D');line=colors.HexColor('#D5E0D7')
styles={
 'body':ParagraphStyle('body',fontName='Chinese',fontSize=10,leading=16,textColor='#243C32',wordWrap='CJK',spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Chinese',fontSize=8.3,leading=12,textColor=muted,wordWrap='CJK',spaceAfter=6),
 'title':ParagraphStyle('title',fontName='ChineseBold',fontSize=23,leading=31,textColor=green,spaceAfter=16),
 'heading':ParagraphStyle('heading',fontName='ChineseBold',fontSize=12,leading=18,textColor=green,spaceBefore=9,spaceAfter=8),
 'cell':ParagraphStyle('cell',fontName='Chinese',fontSize=9,leading=13,textColor='#243C32',wordWrap='CJK'),
 'headcell':ParagraphStyle('headcell',fontName='ChineseBold',fontSize=9,leading=13,textColor=colors.white,wordWrap='CJK')}
story=[];md=[]
def para(t,style='body'):
    story.append(Paragraph(html.escape(str(t)).replace('\n','<br/>'),styles[style]));md.append(str(t)+'\n')
def heading(t):para(t,'heading')
def page(n,title,subtitle):
    if story:story.append(PageBreak());md.append('\n---\n')
    para(f'SPACESCAN TREES  /  DATA AUDIT  /  {n:02d}','small');para(title,'title');para(subtitle,'small');md.append('')
def table(headers,rows,widths):
    cells=[[Paragraph(html.escape(str(v)),styles['headcell' if i==0 else 'cell']) for v in r] for i,r in enumerate([headers]+rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),green),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,line),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5EE'),colors.white])]))
    story.append(t);story.append(Spacer(1,10));md.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('\n',' / ') for x in r)+' |' for r in rows]+[''])
stamp=datetime.fromisoformat(inv['created_at']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M UTC+8')

page(1,'香港数据总盘点报告','盘点时点：'+stamp+'  |  来源只读检查  |  原数据未移动、未改名、未删除')
para('结论：现有香港项目已具备“路线全景 + 历史点云 + 已验收单木配对 + 网络树种参考”的基础；新目录增加了 7 次采集包及 72 份单木分割文件。当前最重要的工作不是立即合并训练，而是补齐新数据的采集批次、树种、影像与单木对应关系。')
heading('核心数据资产')
table(['资产层','已核对规模','当前用途 / 边界'],[
 ['新目录 lidar data/hk','187 文件 / 28.69 GB','7 组采集记录；3 个 LAS 处理区块；72 个单木输出，不等于 72 株已核实树'],
 ['既有 VMMS 三路段',num(sum(stats[k]['files'] for k in ['VMMS_HoManTin','VMMS_TsimShaTsui','VMMS_Stubbs']))+' 文件 / '+gb(sum(stats[k]['bytes'] for k in ['VMMS_HoManTin','VMMS_TsimShaTsui','VMMS_Stubbs']))+' GB','坐标索引覆盖 20,691 个全景帧；另有点云、轨迹及重复表示'],
 ['当前已接受图像-点云配对','95 个视角 / 8 个标签','可追溯配对库；不是 95 株独立树，也不是全部具备完整根冠'],
 ['网络图像与参考库','见第 6 页分来源统计','服务香港项目的树种知识；拍摄地点不一定在香港，不能与街景视角相加']],[110,155,250])
heading('本次“整理”的含义')
para('已生成逻辑分层、文件索引、点云格式检查、单木内容检查、验收状态快照与接入建议；未物理整理源目录，以免破坏既有绝对路径、Demo 引用和训练版本。')
para('统计统一使用十进制 GB（10⁹ 字节）。所选项目及关联目录共扫描 392,505 个文件，约 514.80 GB，包含缓存、实验权重、历史拷贝及位置待定资料；这个数是审计范围的存储规模，不是去重后的香港原始数据集大小。')
heading('优先处理的三件事')
para('1. 核实新 LAS 区块 01/03/04 与采集记录的来源对应，补齐树种与同株影像。\n2. 对 72 个单木输出做独立质量验收，优先检查稀疏实例和根冠边界。\n3. 单独建立新采集的留出验证集，不把已看过的旧回归集当作新的精度证明。')
para('证据：file_inventory.jsonl、inventory.json、content_audit_v2.json；详见第 8 页文件索引。','small')

page(2,'新采集包：7 次记录','位置：TREE/lidar data/hk  |  文件名与 YAML 指向 2026-09-08、09-09、09-11')
rows=[]
for s in sorted(inv['recent_sessions'],key=lambda x:x['id']):
    info=s.get('info.yaml',s.get('start_info.yaml',{}));dense=next(h['points'] for h in s['pointclouds'] if Path(h['path']).name=='map_dense.pcd');v=sum(x.endswith('.insv') for x in s['assets'])
    rows.append([s['id'],info.get('start','不明')[5:16],gb(s['bytes']),f'{dense/1e6:.2f}',str(s.get('trajectory',{}).get('rows','缺失')),str(v)])
table(['批次','开始时间','GB','dense 百万点','pose 行数','INSV 文件'],rows,[70,100,55,85,110,95])
para('7 个采集目录合计 15.77 GB；7 份 map_dense.pcd 的声明点数合计 110,481,010。该合计未作跨批次空间去重，不能解释为独立测量点总数。map.pcd、map_dense.pcd 和 scan_100.pcd 是不同表示或子集，不应累加为新增采集量。')
heading('完整性与配对条件')
para('所有 19 份新 PCD 的二进制大小均与文件头点数/步长一致；7 个 BAG 均具有 ROSBAG V2 文件标识。本轮未完整回放 BAG、校验全部消息或解码 INSV，因此不能据此认定传感器序列无掉帧。')
para('tree982 及 tree9 下 5 个批次共有 6 份 pose.csv，时间严格递增，坐标均为有限值。tree981 缺少 pose.csv 与完整 info.yaml，只有未结束的 start_info.yaml；轨迹可能仍可从 BAG 提取，尚未验证。')
para('只有 tree981、tree982 目录看到了 INSV：各 2 个文件，文件名为同时间的 _00 / _10。它们不自动等同 4 段独立全景视频，需确认相机拼接/同步方式。其余 5 批当前目录未见外置影像，但未排除 BAG 内含影像消息。')
heading('已确认重复')
para('tree981/map.pcd 与 map_dense.pcd 完整 SHA-256 相同，各 1,310,695,967 字节（1.31 GB）。已记录为完全重复文件，未删除；后续统一注册时可保留两个别名、只计一份有效资产。')
para('视频文件名时刻与 YAML 记录存在约 8 小时差异；可能涉及时区，但尚未核实，禁止直接按文件名时间配准。','small')

page(3,'已处理 LAS：3 个区块、72 份单木','这些是分割成果，不是已经人工确认树种的训练真值。')
table(['区块','摘要原始点数','植被点数','单木文件','单木合计点数'],[[b['id'],num(b['semantic_summary']['raw_points']),num(b['semantic_summary']['vegetation_points']),b['individual_tree_files'],num(b['individual_tree_points'])] for b in inv['processed_blocks']],[50,130,130,80,125])
para('72 份单木 LAS 共 4,786,846 点；本轮逐文件读入检查，坐标均为有限值，包围盒与实际坐标一致。此检查确认可读取与几何数值有效，不确认每份只有一棵树、树干/树冠完整或树种正确。')
heading('已有处理链')
para('目录内含地面、植被、其他三类输出，以及 ground、DTM、wood、leaf、understory、segmented_trees、individual_tree_output 等派生产物。三个 summary 记录使用外部路径的 checkpoint_best_3cls.tar（epoch 137）、grid_size 0.06、3 次投票；这是场景分类流程记录，不是当前树种模型的精度报告。')
table(['待核实项','证据与影响'],[
 ['原始文件来源','01 目录未见 01.las，但有 01_full3cls.las 等派生输出；不能说原始链完整。04_summary.json 的 input_las 却指向 02.las，需要确认是否重命名。'],
 ['坐标与文件头','03.las、04.las 文件头 XYZ 最小/最大值均为 0，实际抽样坐标非零。119 份 LAS 中的地理 CRS 信息需以逐文件头记录为准；没有全局配准证明。'],
 ['影像与树种','新目录未见 JPG/PNG 树木照片；3 个 TIF 是 DTM 栅格，不应计为树种识别图像。TreeID 是实例编号，不能当树种标签。']],[105,410])
heading('稀疏实例优先复核')
small=[r for r in audit['single_tree_content'] if r['unique_points']<2048]
para('不足 2048 个独立点的 4 份：'+ '；'.join(f"{r['block']}/{r['id']}（{r['unique_points']} 点）" for r in small)+'。其中 345 点与 246 点的两份不足 512 点。不能靠重复采样补成“高质量 2048 点”。')
para('建议保留完整 LAS 为主档，另生成可追溯的 1024/2048 点副本；缺失尺度、边界、树种或同株影像的样本继续标为待验收。')

page(4,'既有 VMMS：三路段、五个影像流','原始根目录：C:/Users/25041725G/Desktop/receive/vmms')
table(['路段 / 采集批次','文件数','物理容量 GB','索引全景帧'],[
 ['何文田 / 2023-10-27',num(stats['VMMS_HoManTin']['files']),gb(stats['VMMS_HoManTin']['bytes']),'6,088'],
 ['尖沙咀 / 2025-02-10',num(stats['VMMS_TsimShaTsui']['files']),gb(stats['VMMS_TsimShaTsui']['bytes']),'10,720'],
 ['司徒拔道 / 2024-07-25',num(stats['VMMS_Stubbs']['files']),gb(stats['VMMS_Stubbs']['bytes']),'3,883']],[190,100,110,115])
table(['全景流','索引帧数','累计采集轨迹 km'],[[r['stream'],num(r['frames']),f"{r['distance_km']:.4f}"] for r in audit['coordinate_streams']],[285,105,125])
para('五个流累计轨迹约 5.7690 km；它是各序列轨迹长度相加，不是去重后的道路总长。20,691 是坐标索引帧数，也不等于独立树木数或所有图像文件数。')
heading('空间基准与已知故障')
para('既有坐标解算记录采用 Hong Kong 1980 Grid（EPSG:2326）转换 WGS84；来源为插值 INS/平台位置，未应用相机杆臂。不能把这种历史地理定位直接套到新采集的局部 pose 坐标。')
para('45 份地图级点云在坐标索引中登记；还有何文田逐帧彩色云等资料不由该 45 项代表。本轮重新检查 pointcloud_5colored.pcd：实际 32,000,159 字节，头部声明应为 1,615,304,655 字节，明显不完整，不能作为可靠完整云输入。')
heading('范围外 / 待确认')
para('同级 campus、floor6、library 三个目录亦作文件盘点，合计约 4.19 GB；采集地域和与本项目的关系未确证，单列为待确认，未并入三路段数据规模。WHU 专用数据、软件依赖和 Git/worktree 拷贝不作为香港数据统计。')
para('证据：hk-tree-vmms/outputs/coordinates/coordinate_summary.json；本轮 file_inventory.jsonl；异常 PCD 实时文件头检查。','small')

page(5,'标注与单木配对：区分接受、草稿和几何验收','当前配对目录：outputs/hk_point_assist_20260911_v1/catalog.json')
names={'Livistona chinensis':'蒲葵','Wodyetia bifurcata':'狐尾椰子','Ficus microcarpa':'榕树（细叶榕）','Delonix regia':'凤凰木','Aleurites moluccana':'石栗','Araucaria columnaris(Araucaria heterophylla)':'南洋杉（历史合并类）','Leucaena leucocephala':'银合欢','Albizia lebbeck':'大叶合欢'}
table(['现有标签','学名 / 历史映射','接受视角数'],[[names[k],k,v] for k,v in audit['current_pair_species'].items()],[125,310,80])
para('总计 95 个接受配对视角、8 个标签。版本选择保留用户接受记录，但不代表 95 株独立树、不代表全根冠完整、也不表示这些标签都已在本轮独立复核。榕树显示采用 Ficus microcarpa；南洋杉保留历史合并类信息，不擅自拆成两个真值类。')
heading('最新人工状态，不能只看旧汇总文件')
para('全景正式状态：389 帧，其中 200 帧 annotated、146 帧 no_tree、43 帧 unusable，包含 614 条标签记录。另有 1,065 帧草稿、4,653 条标签；草稿可能包含模型预标注和重叠正式帧，不能直接相加或全部视为人工真值。')
g=audit['current_geometry_reviews']['flags']
para(f"点云辅助模块有 16 条独立几何核验记录：同株确认 {g['same_tree']}、无明显混入 {g['isolated_tree']}、基部可见 {g['base_visible']}、冠部覆盖 {g['crown_complete']}、另行核实树种 {g['species_verified']}。这些是用户勾选数，不是算法精度；未勾选意味着未确认，不等于否定旧树种标签。")
heading('现阶段训练 / 验证边界')
para('既有五类试验留出集仅 10 个视角、7 个保守空间组，真值只覆盖石栗、大叶合欢、南洋杉；已查看过，只能作为回归集。近期点云辅助受控实验存在条件性收益，但不构成全部 8 类或新采集数据的普适精度证据。')
para('榕树覆盖显示 v3 为 16 个配对修正旧“仅补幹”投影，并优化采样；原全量几何与权重不变，不能把显示修复算成新增训练样本。','small')

page(6,'网络图像、参考图与批处理结果','训练素材、演示参考和自动预测必须分开管理。')
table(['数据源 / 分区','本轮核对数量','状态说明'],[
 ['iNaturalist max150','20,448 图像文件 / 150 个目录','按实际图像路径统计，未做全库跨来源去重；目录数量不等于最终模型类别数'],
 ['inaturalist_dated','13,080 图像文件','含元数据关联资料；与其他网络库可能重复，不能直接累加为独立样本'],
 ['旧 dataset','373 图像 / 12 个目录','历史素材，继续保留，不自动归入新训练清单'],
 ['香港植物标本室','197 图像 / 28 个源类群','SQLite 当前接受 131、拒绝 66；旧 summary 的 accepted=0 已过时'],
 ['王棕 / 狐尾椰补充','117 条整理候选','SQLite 当前接受 26、拒绝 13；其余 78 条未见 decisions 行，不能视为接受'],
 ['八常见树种权威补充','45 张清单记录 / 8 类','含 18 次新下载和既有资料引用；不是新 45 个独立采集样本'],
 ['当前自动参考图库','293 条 / 147 个标签','146 类达到两张，1 类只有一张，1 类暂无；训练资格为 false']],[125,155,235])
para('参考图库缺口：Elaeocarpus sylvestris 现有 1 张；Photinia benthamiana 现有 0 张。参考图允许花叶细节填补整株不足，但不能当同株多视角，也不能自动用于模型训练。')
heading('已有 20 米全景识别成果')
para('已处理 343 个锚点帧、1,009 个相机帧，原始候选 799 个，232 条暂定多视角关联。建筑误检过滤、Google 街景核验、暗区/超分辨率对照与人工评估另存版本。上述数量是预测和处理数量，不是准确率，展示过滤后的记录也不是完整真值集。')
heading('授权与训练安全')
para('网络照片服务于香港树种项目，但拍摄地点不一定在香港。保留来源、原水印、作者和授权字段；图像质量接受不是版权许可。每次训练仍须先备份初始化与部署权重、冻结接受清单与分组划分，不自动纳入拒绝、未审和参考专用图。')
para('模型和实验目录属于数据的消费端：YOLO、BioCLIP、点云编码器及各分类头需与标签映射绑定。模型类别数不能用素材文件夹数量代替，实验缓存也不能算新采集数据。','small')

page(7,'问题清单与接入顺序','本轮仅报告和建立索引；以下修复与训练尚未执行。')
table(['优先级','问题','建议动作 / 验收标准'],[
 ['P0','新数据来源链不完整','建立 session -> LAS 区块 -> TreeID -> 图像 -> 树种表；核实 04/02 名称和缺失 01 原档'],
 ['P0','新旧坐标不能直接混用','记录坐标单位、CRS、轴向、尺度、时间基准及相机-LiDAR 外参；配准误差必须有检查记录'],
 ['P1','LAS 包围盒异常','在新派生副本修复 03/04 文件头，完整重算边界并对照原点数；不得覆盖原件'],
 ['P1','72 份单木质量未验收','检查孤立单木、根冠覆盖、邻树/地面混入；优先 4 个低点数样本，拒绝靠复制点提高点数'],
 ['P1','部分传感器资料待恢复','从 tree981 BAG 查找轨迹；确认其他批次是否另有影像；核对 INSV 双文件和时间差'],
 ['P2','重复及陈旧状态','给 tree981 两个相同地图登记同一内容 ID；读取 SQLite 当前 decisions，不用旧下载摘要覆盖验收结果'],
 ['P2','旧地图不完整','隔离使用 pointcloud_5colored.pcd，优先查找完整同源版本；不要误当正常输入']],[45,135,335])
heading('推荐落地的统一注册表')
para('每个单木版本至少记录：asset_id、survey_id、tree_group_id、source_path、SHA-256、采集时间、坐标/单位、图像与点云路径、物种原标签与统一映射、几何质量、人工验收、权利状态、split_group、处理脚本版本。原件只读；新版本和拒绝记录分开。')
heading('先验收，再验证，再接入 Demo')
para('第一步把 72 个单木建立带图像的验收页；第二步按独立树木和采集批次冻结训练/验证/测试；第三步在相同样本上比较图像单源、点云单源与双源，分别报告天然低质量和受控退化；第四步通过质量门槛并经用户确认后，才将新版本接入正式识别分支。')
para('建议先选“来源明确 + 有同株影像 + 根冠可辨”的样本做首批，不按模型是否预测正确挑选测试样本。')

page(8,'可追溯目录与统计口径','本报告与明细构成逻辑整理成果，不改变现有项目结构。')
table(['逻辑分层','真实位置（相对 TREE，除 VMMS）'],[
 ['近期采集及处理成果','lidar data/hk/；其中 tree981、tree982、tree9/* 为采集目录，data/data/{01,03,04} 为处理区块'],
 ['历史原始数据','../vmms/{2023-10-27_hewentian,20250210-jianshazui,Stubbs Road_2024-07-25_HyD_collect}'],
 ['地理索引与人工标签','hk-tree-vmms/outputs/coordinates/；hk-tree-vmms/derived/exhaustive_species_frame_labeler/runtime/'],
 ['单木配对和当前入口','hk-tree-vmms/derived/ 下各版本；outputs/hk_point_assist_20260911_v1/catalog.json'],
 ['网页图像与参考','external_datasets/；dataset/；参考专用清单为 spacescan_whole_tree_references_20260910_v2/mixed_v3/'],
 ['预测/实验/部署','outputs/vmms20m_teacher_20260909_v1/；outputs/distillation/；spacescan_trees_demo_v1/']],[120,395])
heading('本次报告附件')
para('reports/hk_data_inventory_20260912_v1/\n• REPORT.md：可检索的报告文字版。\n• file_inventory.jsonl：392,505 条文件级位置、大小、类型与修改时间。\n• inventory.json：目录统计、138 份点云头部、7 组轨迹记录、来源汇总与验收快照。\n• content_audit_v2.json：72 份单木全内容检查、重复哈希、LAS 抽样与当前配对/参考统计。\n• content_audit.json：初次内容审计，目录提取口径已由 v2 更正；分析请以 v2 为准。')
heading('校验程度与不能据此推断的结论')
para('完成：源目录文件元数据盘点；新 19 份 PCD 结构大小核对；119 份 LAS 头部；72 份单木完整坐标读取与哈希；两份大型 LAS 分段抽样；6 份轨迹全行检查；指定重复地图完整哈希；SQLite 只读查询。')
para('未完成：全部原始大文件哈希和完整回放、所有网络图去重、全部语义/实例/树种标注复核、新旧采集配准、全部 LAS 全量内容审查和完整树高认证。因此“可读取”“接受配对”“独立树木”“训练合格”是不同层级，不能互相替代。')
para('只读盘点无文件读取错误。来源可能随之后的用户验收改变，本报告是盘点时点快照；后续训练应再次锁定最新人工决定，而非将此快照永久当作当前状态。','small')

def footer(c,doc):
    w,h=A4;c.setStrokeColor(line);c.line(40,38,w-40,38);c.setFont('Chinese',8);c.setFillColor(muted);c.drawString(40,25,'SpaceScan Trees | 香港数据资产盘点 | 2026-09-12');c.drawRightString(w-40,25,str(doc.page))
PDF.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=50,title='香港数据总盘点报告 - SpaceScan Trees',author='SpaceScan Trees').build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'REPORT.md').write_text('# 香港数据总盘点报告\n\n'+'\n'.join(md),encoding='utf-8')
print(PDF)
