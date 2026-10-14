import copy
import unittest
from fusion.doc import apply, invert_keymap, rename_keys


def fixture():
    stage = {'components':[{'id':'a','type':'c','bbox_json':[0,0,10,10]},
                           {'id':'b','type':'c','bbox_json':[30,0,40,10]}]}
    assigned = {'comps':{'a':{'designator':{'text':'C14'}},'b':{'designator':{'text':'C24'}}}}
    doc = {'components':{'C24':{'type':'c','Name':'C24','bbox':[0,0,10,10]},
                         'C34':{'type':'c','Name':'C34','bbox':[30,0,40,10]}},
           'pins':{'C24':{'pin_1':{'pinname':'1'}},'C34':{'pin_2':{'pinname':'2'}}},
           'nets':{'n':{'hyperGraph':'(C24.1,C34.2)'}}}
    return stage,assigned,doc


class NonICDocumentIdentityTests(unittest.TestCase):
    def test_rebuilt_key_reuse_is_resolved_by_body_identity(self):
        stage, assigned, doc = fixture()
        self.assertEqual(invert_keymap(doc, stage, assigned), {'a':'C24','b':'C34'})

    def test_already_rebuilt_names_are_not_applied_to_another_component(self):
        stage, assigned, doc = fixture(); old=copy.deepcopy(doc)
        sol={'assign':{'comp.a.key':('C24',.9,'geo:axis'), 'comp.b.key':('C34',.8,'ocr:designator')}}
        result,report=apply(doc,stage,assigned,sol)
        self.assertEqual(result,old)
        self.assertEqual(report['keys_same'],2)
        self.assertEqual(doc,old)

    def test_rejected_target_move_does_not_allow_overwrite_of_its_source(self):
        doc={'components':{k:{'type':'r','Name':k}for k in ['R44','R45','R46']},
             'pins':{},'nets':{}}
        old=copy.deepcopy(doc)
        self.assertEqual(rename_keys(doc,{'R44':'R45','R45':'R46'}),{})
        self.assertEqual(doc,old)

    def test_swap_rewrites_network_tokens_once(self):
        doc={'components':{'R44':{'type':'r','Name':'R44'},'R45':{'type':'r','Name':'R45'}},
             'pins':{'R44':{'pin_1':{}},'R45':{'pin_2':{}}},
             'nets':{'n':{'hyperGraph':'(R44.1,R45.2)'}}}
        rename_keys(doc,{'R44':'R45','R45':'R44'})
        self.assertEqual(doc['nets']['n']['hyperGraph'],'(R45.1,R44.2)')
        self.assertIn('pin_1',doc['pins']['R45'])
        self.assertIn('pin_2',doc['pins']['R44'])

    def test_chain_preserves_each_original_network_reference(self):
        doc={'components':{'R44':{'type':'r','Name':'R44'},'R45':{'type':'r','Name':'R45'}},
             'pins':{},'nets':{'n':{'hyperGraph':'(R44.1,R45.2,R444.3)'}}}
        rename_keys(doc,{'R44':'R45','R45':'R46'})
        self.assertEqual(doc['nets']['n']['hyperGraph'],'(R45.1,R46.2,R444.3)')
        self.assertEqual(len(doc['components']),2)


if __name__=='__main__':unittest.main()
