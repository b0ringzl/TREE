import copy,json,re,unittest
from demo_locale import present_result,text,catalog_ui,MODELS,species_name

class LocaleTests(unittest.TestCase):
    def test_unmapped_species_uses_name_only(self):
        self.assertEqual(species_name('Bauhinia x blakeana'),'Bauhinia x blakeana')
        self.assertEqual(species_name('Bauhinia variegata'),'Bauhinia variegata')
        self.assertEqual(species_name('Livistona chinensis'),'Livistona chinensis / 蒲葵')
    def sample(self):
        b={'label':'榕树','display_name':'榕树','score':.7,'source':'香港模型','reasons':['输入质量严重不足'],'top3':[{'species':'榕树','name':'榕树','score':.7}]}
        return {'domain':'hongkong_vmms','decision':b,'results':{'image':b},'quality':{},'warnings':['香港三类为实验'], 'occlusion_grid':[],'point_preview':[],'elapsed_seconds':1,'device':'cpu','reference':{'truth':'榕树','split':'train','sample_id':'wuhan_041','correct':True}}
    def test_presentation_preserves_raw(self):
        d=self.sample();before=copy.deepcopy(d);p=present_result(d)
        self.assertEqual(d,before);self.assertEqual(p['decision']['score'],.7)
        self.assertNotRegex(json.dumps(p,ensure_ascii=False),r'香港|武汉|武漢|[Hh]ong.?[Kk]ong|[Ww]uhan|榕树')
        self.assertIn('榕樹',p['decision']['display_name']);self.assertNotIn('sample_id',p['reference'])
    def test_diagnostics(self):
        for raw in ['请至少提供图像或点云。','NPZ需要points_xyz、points或xyz数组。','图像过暗','前两名差值低于演示阈值 0.15','至少一种输入质量严重不足']:
            localized=text(raw);self.assertRegex(localized,'[A-Za-z]');self.assertRegex(localized,'[\u4e00-\u9fff]');self.assertNotIn('Additional diagnostic',localized)
        self.assertIn('fusion is disabled',text('至少一种输入质量严重不足'))
    def test_models(self):
        for name,scope in MODELS.values():self.assertNotRegex(name+scope,r'香港|武汉|武漢|[Hh]ong.?[Kk]ong|[Ww]uhan')

if __name__=='__main__':unittest.main()
