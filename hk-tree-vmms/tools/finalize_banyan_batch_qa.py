"""Persist assistant contact-sheet screening, separately from human reviews."""
from collections import Counter
import hashlib
import numpy as np
from audit_hk_metric_height import DATA,read,write

OUT=DATA/'banyan_tracking_batch_v2'
# Queue indices were inspected on qa_sheet_1..5, not selected by prediction.
PREFER={1,2,4,5,6,8,11,13,17,20,21,22,27,30}
REASONS={3:'悬空/横向结构，未见可信主干连接',7:'冠下出现长水平面，疑混入地表',
         9:'基底横向结构疑混入花坛',10:'冠下较长水平条带，继续清洗',12:'冠下跨多干，疑背景/邻树混入',
         14:'横向冠下结构及多干归属不清',15:'仅细长局部，不能代表该粗干树整体',
         16:'冠下多处分散支持，需排查邻干和花坛',18:'基底横向条带疑背景',19:'基底平面疑花坛残留',
         23:'冠下存在大面积非树状结构',24:'横向区域且缺基干，疑定位/地面错误',
         25:'冠下横向连片，根颈及单木归属不明确',26:'冠下大面积平面结构，继续返工',
         28:'基底横向条带，继续清洗',29:'明显横向基底，疑混入花坛'}

def main():
    queue=read(OUT/'visual_queue.json');qa=read(OUT/'agent_visual_qa.json')
    manifest={r['id']:r for r in read(DATA/'manifest.json')}
    accepted=read(DATA/'banyan_tracking_pilot_v1/reviews.json')
    kept=[manifest[i] for i,v in accepted.items() if v['decision']=='accept']
    for r in queue:
        idx=r['index'];ident=r['id'];entry={'disposition':'review_candidate' if idx in PREFER else 'needs_rework',
          'note':'投影可见主干/分叉实测支持，供进一步复核；根颈、邻干及完整性未认证。' if idx in PREFER else REASONS[idx],
          'source':'assistant_contact_sheet_visual_screening_not_human_acceptance', 'sheet_index':idx}
        if idx in PREFER:
            source=manifest[ident];axis=np.array(source['quality']['axis_local_xy'])
            near=[s for s in kept if s['survey']==source['survey'] and np.linalg.norm(axis-s['quality']['axis_local_xy'])<3]
            if near:
                entry.update(disposition='alternate_view',note='距已有优先/接受视角树轴不足3米，暂不重复交验；未继承任何接受状态。',nearby_reference_ids=[s['id'] for s in near])
            else:kept.append(source)
        qa[ident]=entry
    write(OUT/'agent_visual_qa.json',qa)
    results=read(OUT/'results.json');scope=read(OUT/'scope.json')
    report={'processed_pairs':len(results),'raw_status_counts':dict(Counter(r['status'] for r in results)),
            'screening_dispositions':dict(Counter(r['disposition'] for r in qa.values())),
            'visual_queue_count':len(queue),'deferred_frame_cloud_pairs':len(scope['deferred_no_dense_cache']),
            'accepted_pilot_preserved':sum(v['decision']=='accept' for v in accepted.values()),
            'pilot_review_hash_unchanged':hashlib.sha256((DATA/'banyan_tracking_pilot_v1/reviews.json').read_bytes()).hexdigest()==scope['pilot_review_hash'],
            'model_changed':False,'all_new_results_unaccepted':not (OUT/'reviews.json').exists()}
    write(OUT/'delivery_summary.json',report);print(report)

if __name__=='__main__':main()
