"""Bilingual, geography-neutral presentation. Never changes inference labels."""
import copy,re

MODELS={
 'hongkong':('General image · 19 classes / 通用影像 · 19 類','Image recognition; point clouds support quality checks and preview only. / 影像識別；點雲僅支援品質檢查與預覽。'),
 'wuhan':('Paired multimodal · 17 classes / 配對多模態 · 17 類','Image, point-cloud and trained fusion models; confirm same-tree pairing. / 影像、點雲與已訓練融合模型；請確認同木配對。'),
 'hongkong_vmms':('Street-tree multimodal · 3 classes / 街景多模態 · 3 類','Banyan, fan palm and foxtail palm; experimental transfer heads. / 榕樹、蒲葵與狐尾椰子；實驗性遷移分類頭。'),
 'hongkong_vmms_five':('Extended multimodal · 5 classes / 擴展多模態 · 5 類','Five-class experiment; image-first output, separate fusion candidates and conflict alerts. / 五類實驗；影像優先輸出，融合候選獨立展示並提示分歧。')}
TRAD={
 'Ravenala madagascariensis':'旅人蕉','Araucaria columnaris(Araucaria heterophylla)':'南洋杉（原訓練合併類）','Livistona chinensis':'蒲葵',
 'Thevetia peruviana':'黃花夾竹桃','Casuarina equisetifolia':'木麻黃','Michelia x alba':'白蘭','Macaranga tanarius':'血桐',
 'Eucalyptus urophylla':'尾葉桉','Machilus velutina':'絨毛潤楠','Plumeria rubra':'雞蛋花','Elaeocarpus hainanensis':'水石榕',
 'Podocarpus macrophyllus':'羅漢松','Lagerstroemia speciosa':'大花紫薇','Caryota mitis':'短穗魚尾葵','Reevesia thyrsoidea':'梭羅樹',
 'Terminalia catappa':'欖仁樹','Celtis sinensis':'朴樹','Diospyros eriantha':'小果柿','Roystonea regia':'王棕','Cedrus deodara':'雪松',
 'Malus x micromalus':'西府海棠','Metasequoia glyptostroboides':'水杉',"Prunus cerasifera 'atropurpurea'":'紫葉李',
 'Photinia x fraseri':'紅葉石楠','Zelkova serrata':'櫸樹','Koelreuteria paniculata':'欒樹','Magnolia grandiflora':'廣玉蘭',
 'Lagerstroemia indica':'紫薇','Ginkgo biloba':'銀杏','Osmanthus fragrans':'桂花','Acer pictum subsp. mono':'五角楓',
 'Platanus x acerifolia':'二球懸鈴木','Prunus serrulata':'櫻花','Acer palmatum var. atropurpureum':'紅楓','Sophora japonica':'國槐',
 'Ligustrum lucidum':'女貞','榕树':'榕樹','Wodyetia bifurcata':'狐尾椰子','Delonix regia':'鳳凰木',
 'Aleurites moluccana':'石栗','Leucaena leucocephala':'銀合歡','Albizia lebbeck':'大葉合歡','Cinnamomum camphora':'樟樹'}
MODES={'image':'Image / 影像','point':'Point cloud / 點雲','fusion':'Fusion / 融合'}
def species_name(name):
    if name=='Unknown':return 'Unknown / Review required / 未知／需複核'
    # Missing translations must not appear as a fabricated species name.
    latin='Banyan group' if name=='榕树' else name
    return latin+' / '+TRAD[name] if name in TRAD else latin
def scientific(name):return 'Banyan group / 榕樹類' if name=='榕树' else name
def split_name(split):
    if 'training' in split or split=='train':return 'Training example — not a test / 訓練示例，並非測試'
    if 'weak' in split:return 'Provisional-label holdout / 暫定標籤留出樣本'
    if 'pilot' in split:return 'Small pilot holdout / 小樣本實驗留出集'
    if split in ('test','heldout','held_out_test'):return 'Held-out example / 留出示例'
    if 'probe' in split:return 'Out-of-catalog probe / 類別範圍外探測'
    return 'Reference example / 參考示例'

# Keywords cover equivalent diagnostics from all four existing model domains.
RULES=[
 ('分辨率不足','Resolution too low; provide a clear crop at least 224 pixels wide and high. / 解像度不足；請提供寬高至少 224 像素的清晰裁切圖。'),
 ('图像过暗','Image too dark; increase exposure and avoid backlighting. / 影像過暗；請提高曝光並避免逆光。'),
 ('高光丢失','Highlights are clipped; reduce exposure to retain leaf detail. / 高光細節流失；請降低曝光以保留葉片細節。'),
 ('图像疑似模糊','Image may be blurred; refocus and stabilize the camera. / 影像可能模糊；請重新對焦並穩定相機。'),
 ('宽幅图像','Wide image may contain multiple trees; crop one target. / 寬幅影像可能包含多棵樹；請先裁切單一目標。'),
 ('接近平面或细线','Cloud is nearly planar or linear; remove walls and recover the tree crown. / 點雲近似平面或細線；請排除牆面並補充樹冠。'),
 ('独立点数偏少','Few distinct points; add real scanning coverage rather than duplicated points. / 獨立點數偏少；請補充實際掃描覆蓋，不要複製點數。'),
 ('重复点比例偏高','Many duplicate points; deduplicate and add new scan views. / 重複點比例偏高；請去重並補充新掃描視角。'),
 ('高度跨度很小','Small vertical span; check upright Z orientation and crown coverage. / 垂直跨度偏小；請檢查 Z 軸朝向及樹冠覆蓋。'),
 ('不在本demo','Top candidate is outside this model catalog. / 首選候選不在此模型清單內。'),
 ('输入质量','Input quality is insufficient; review or improve the source. / 輸入品質不足；請複核或改善來源資料。'),
 ('至少一种输入质量','One input is low quality; fusion is disabled and a usable source is preferred. / 其中一種輸入品質不足；已停用融合並優先採用可用單源。'),
 ('至少一源质量','One input is low quality; fusion is disabled and a usable source is preferred. / 其中一種輸入品質不足；已停用融合並優先採用可用單源。'),
 ('未确认','Same-tree pairing is unconfirmed; sources are classified separately without fusion. / 尚未確認同木配對；各來源獨立識別，不進行融合。'),
 ('首选树种不一致','Image and point-cloud candidates disagree; check pairing and tree boundaries. / 影像與點雲首選樹種不一致；請檢查配對及單木邊界。'),
 ('两源首选一致','Both sources agree, but agreement does not prove correctness. / 兩個來源的首選一致，但一致並不代表正確。'),
 ('融合结果单独','Fusion is shown separately; it has not demonstrated an advantage over images. / 融合結果獨立展示；尚未證明優於影像單源。'),
 ('首选属于未独立','This candidate class has no independent evaluation; human review is needed. / 此候選類別尚無獨立評估；需要人工複核。'),
 ('扩展五类为实验','Five-class experiment: flame tree and white leadtree lack independent evaluation; point-cloud reliability is unverified. / 五類實驗：鳳凰木及銀合歡尚無獨立評估，點雲識別可靠性未驗證。'),
 ('仅适用于已裁剪','Single-tree input only; this is not automatic multi-tree panorama detection. / 僅支援單木輸入；並非全景多樹自動偵測。'),
 ('三类为实验','Three-class transfer experiment; transferred labels do not establish independently reviewed accuracy. / 三類遷移實驗；轉移標籤不能證明獨立人工複核精度。'),
 ('点云可预览','Point clouds support preview and quality checks; classification uses the image model. / 點雲支援預覽及品質檢查；分類結果由影像模型提供。'),
 ('点云模型仅在','This point-cloud model does not cover the selected image catalog. / 此點雲模型未涵蓋所選影像類別清單。'),
 ('特征与训练','Input features differ substantially from the training references. / 輸入特徵與訓練參考差異較大。'),
 ('质量相对较好','This is a quality-limited reference; inspect completeness and visible coverage. / 此參考樣本品質受限；請檢查完整性與可見覆蓋。'),
 ('至少提供','Provide at least one image or point cloud. / 請至少提供一張影像或一份點雲。'),
 ('图像过大','Image exceeds 40 million pixels; resize before uploading. / 影像超過四千萬像素；請縮小後上傳。'),
 ('单文件限80MB','Each file must be at most 80 MB; crop a single tree first. / 每個檔案上限為 80 MB；請先裁切單木。'),
 ('NPZ需要','NPZ requires points_xyz, points or xyz. / NPZ 需要 points_xyz、points 或 xyz 陣列。'),
 ('超过200万','Cloud exceeds 2 million points; extract a single tree first. / 點雲超過二百萬點；請先提取單木。'),
 ('支持点云格式','Supported clouds: NPZ, NPY, XYZ, TXT, CSV, LAS, LAZ. / 支援點雲格式：NPZ、NPY、XYZ、TXT、CSV、LAS、LAZ。'),
 ('需要N×3','Point cloud requires an N × 3 XYZ array. / 點雲需要 N × 3 的 XYZ 陣列。'),
 ('少于32','Fewer than 32 points; provide more real measurements. / 點數少於 32；請補充實測資料。'),
 ('NaN或无穷','Cloud contains NaN or infinite values; clean the coordinates first. / 點雲含 NaN 或無限值；請先清理座標。'),
 ('全部重合','All point coordinates coincide; this cloud cannot be classified. / 所有點座標重合；無法識別此點雲。'),
 ('模型范围','Select a valid model scope. / 請選擇有效的模型範圍。'),
 ('映射','Model label mapping failed validation. / 模型標籤映射驗證失敗。'),
 ('顺序','Model class ordering failed validation. / 模型類別順序驗證失敗。'),
 ('不一致','Model or encoder versions do not match; verify the local assets. / 模型或編碼器版本不一致；請檢查本地資產。'),
 ('权重','Required experimental weights are unavailable; verify the local installation. / 所需實驗權重不可用；請檢查本地安裝。'),
 ('样本不存在','The requested example is unavailable. / 所選示例不存在。'),
 ('无效模式','Invalid input mode. / 輸入模式無效。')]

def text(value):
    value=str(value)
    if not value:return ''
    if ('最高' in value or '分数' in value) and ('低于' in value):
        number=re.findall(r'0\.\d+',value);threshold=number[0] if number else '—'
        return f'Top score is below the uncalibrated demo threshold {threshold}. / 最高分數低於未校準的演示門檻 {threshold}。'
    if '前两名' in value and ('差值' in value or '分数差' in value):
        return 'Top-two margin is below the uncalibrated demo threshold 0.15. / 前兩名分數差低於未校準的演示門檻 0.15。'
    # More specific quality fallback must precede the generic quality phrase.
    ordered=sorted(RULES,key=lambda x:('至少' not in x[0],))
    for key,result in ordered:
        if key in value:return result
    if not re.search('[\u4e00-\u9fff]',value):
        if re.search(r'hong\s*kong|wuhan',value,re.I):return 'Model diagnostic; review the local setup. / 模型診斷提示；請檢查本地設定。'
        return value+' / 技術診斷訊息'
    return 'Additional diagnostic: review input quality and model compatibility. / 額外診斷：請複核輸入品質及模型相容性。'

def present_result(data):
    d=data;model=MODELS[d['domain']][0]
    def branch(value,mode):
        label=value['label'];result={**value}
        result['label']=scientific(label);result['display_name']=species_name(label)
        if '分歧' in value.get('display_name',''):result['display_name']='Source disagreement / Review required / 來源分歧／需複核'
        result['source']=model+' · '+MODES[mode]
        result['reasons']=[text(x) for x in value.get('reasons',[])]
        result['score_kind']='Uncalibrated model score — not accuracy / 未校準模型分數，並非正確率'
        if 'reliability' in result:
            result['reliability']='No independent class evaluation / 尚無獨立類別評估' if value['reliability']=='no_independent_class_evaluation' else 'Small holdout only / 僅小樣本留出驗證'
        result['top3']=[{**v,'species':scientific(v['species']),'name':species_name(v['species'])} for v in value.get('top3',[])]
        if result.get('candidate_label'):result['candidate_label']=scientific(result['candidate_label'])
        return result
    outputs={m:branch(v,m) for m,v in d['results'].items()}
    mode=d.get('decision_source_mode') or next((m for m,v in d['results'].items() if v.get('source')==d['decision'].get('source')),next(iter(outputs)))
    quality=copy.deepcopy(d['quality'])
    for q in quality.values():
        q['warnings']=[text(x) for x in q.get('warnings',[])]
        q['scope']='Heuristic quality checks; not proof of species, pairing or completeness. / 啟發式品質檢查，不能證明樹種、配對或完整性。'
    evidence=['Model features are computed from the input; reference labels are attached only after inference. / 模型特徵由輸入計算；參考標籤只在推理完成後附加。']
    if 'image' in outputs:evidence.append(f"Image top-two margin: {outputs['image'].get('margin','—')}; not a calibrated probability. / 影像前兩名分數差：{outputs['image'].get('margin','—')}；並非校準機率。")
    for m,v in outputs.items():
        if 'feature_distance' in v:evidence.append(f"{MODES[m]} · Distance to training references: {v['feature_distance']}; a model-space diagnostic, not species proof. / 與訓練參考的特徵距離：{v['feature_distance']}；屬模型空間診斷，並非樹種證明。")
    if 'point' in quality:
        q=quality['point'];evidence.append(f"Distinct points: {q['unique_count']}; vertical-to-width ratio: {q['height_to_width_ratio']}. Not a verified tree height. / 獨立點數：{q['unique_count']}；高寬比：{q['height_to_width_ratio']}。並非已驗證樹高。")
        if q.get('inference_points'):evidence.append(f"Encoder input: {q['inference_points']} real points; sparse assets are not padded. / 編碼器輸入：{q['inference_points']} 個實測點；稀疏資產不補點。")
    if d['occlusion_grid']:evidence.append('Occlusion sensitivity shows changes in model score, not botanical evidence. / 遮擋敏感度顯示模型分數變化，並非植物學證據。')
    if 'fusion' in outputs:evidence.append('The fusion head uses paired learned features, not a lookup of the example label. / 融合分類頭使用配對的學習特徵，並非查詢示例標籤。')
    if d.get('experimental'):evidence.append('Frozen image / point encoders with experimental transfer heads; not end-to-end fine-tuning. / 凍結影像／點雲編碼器並使用實驗遷移分類頭；並非端到端微調。')
    limitations=['Single-tree classification only; crop or extract the target first. / 僅支援單木分類；請先裁切或提取目標。',
      'Scores and rejection thresholds are uncalibrated; out-of-catalog species may be misclassified. / 分數及拒識門檻未校準；範圍外物種仍可能誤判。',
      'Model-domain results do not establish accuracy for an arbitrary street scene. / 模型資料域的結果不能證明任意街景的識別精度。']
    if d['domain']=='hongkong_vmms_five':limitations.append('Flame tree and white leadtree lack independent evaluation; cloud reliability is unverified. / 鳳凰木及銀合歡尚無獨立評估；點雲可靠性未驗證。')
    result={'model':model,'decision':branch(d['decision'],mode),'results':outputs,'quality':quality,
      'warnings':[text(x) for x in d['warnings']],'evidence':evidence,'limitations':limitations,
      'occlusion_grid':d['occlusion_grid'],'point_preview':d['point_preview'],'elapsed_seconds':d['elapsed_seconds'],'device':d['device'],
      'experimental':d.get('experimental',False),'language':'en + zh-Hant'}
    if d.get('reference'):
        r=d['reference'];result['reference']={'truth':species_name(r['truth']),'split':split_name(r['split']),'correct':r['correct']}
    return result

def attach(data):data['presentation']=present_result(data);return data
def catalog_ui(raw):
    groups=[]
    for key,(name,scope) in MODELS.items():
        rows=[]
        for item in raw.get(key,[]):
            taxon=item['scientific_name'];rows.append({'species':scientific(taxon),'name':species_name(taxon),'f1':item.get('f1'),
                'support':item.get('support'),'accepted_views':item.get('accepted_views'),'modalities':item['modalities'],
                'validation':'No independent evaluation / 尚無獨立評估' if item.get('validation_scope')=='no_independent_class_evaluation' else
                  'Small pilot only / 僅小樣本實驗' if key=='hongkong_vmms_five' else 'Provisional-label experiment / 暫定標籤實驗' if key=='hongkong_vmms' else 'Model-domain validation / 模型資料域驗證'})
        groups.append({'key':key,'name':name,'scope':scope,'items':rows})
    return {'models':groups}
