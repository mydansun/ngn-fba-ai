"""Compare frozen production predictions with retrained models on aligned held-out documents."""
import argparse
from collections import Counter
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import time
import cv2
import torch


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def canon(value):return ''.join(str(value or '').split())


def score(gold,pp):
    def quantity(value):
        try:return int(value)
        except (TypeError,ValueError):return None
    expected=Counter((canon(i['sku']),quantity(i.get('qty'))) for i in gold['items'])
    actual=Counter((canon(i['sku'].get('clean')),quantity(i['qty'].get('value'))) for i in pp['items'])
    return {'track':canon(gold['tracking_no'])==canon(pp['track'].get('clean')),
            'recipient':canon(gold['recipient']).casefold().rstrip(',')==canon(pp['recipient'].get('clean')).casefold().rstrip(','),
            'skus':Counter(k[0] for k in expected.elements())==Counter(k[0] for k in actual.elements()),
            'items':expected==actual}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--clean-run',type=Path,required=True);p.add_argument('--clean-code',type=Path,required=True)
    p.add_argument('--postprocess',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layout-code',type=Path,default=Path(__file__).resolve().parents[1]/'packages/layout/ngn_fba_layout/runtime.py')
    p.add_argument('--split',default='test',choices=['test','validation','llm_holdout'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    layout=module('layout_runtime',a.layout_code)
    clean=module('clean_runtime',a.clean_code)
    post=module('legacy_postprocess',a.postprocess)
    runner=layout.LayoutLMRunner(a.model,512,128,3,torch.device('cuda'),4,True)
    cleaners={}
    for kind in ('track','recipient','sku'):
        root=a.clean_run/kind;vocab=clean.load_vocab(root/'vocab.json')
        checkpoints=[root/'model.ckpt'] if (root/'model.ckpt').is_file() else list((root/'checkpoints').glob('best-*.ckpt'));assert len(checkpoints)==1
        model,cfg=clean.load_model_from_ckpt(checkpoints[0],vocab,torch.device('cuda'))
        cleaners[kind]=clean.Cleaner(model,vocab,cfg,torch.device('cuda'))
    def cleaned(pp):
        pp=copy.deepcopy(pp)
        for kind in ('track','recipient','sku'):
            fields=[i['sku'] for i in pp['items']] if kind=='sku' else [pp[kind]]
            if not fields:continue
            values,_=cleaners[kind].clean_batch_with_meta([f['raw'] for f in fields],.5,512,256)
            for field,value in zip(fields,values):field['clean']=value
        return pp
    docs={d['id']:d for d in json.loads((a.data/'prepared/layout.json').read_text()) if d['split']==a.split}
    cases=[]
    with gzip.open(a.data/'records.jsonl.gz','rt') as f:
        next(f)
        for line in f:
            record=json.loads(line);doc=docs.get(record['_id'])
            if doc is None:continue
            directory=a.data/'layout'/a.split
            ocr=json.loads((directory/(doc['id']+'-ocr.json')).read_text())
            image=cv2.imread(str(directory/(doc['id']+'.jpeg')))
            assert image is not None
            started=time.perf_counter()
            result,_=runner.infer_with_meta(image,*layout.build_items_from_ocr_res(ocr,image))
            new=cleaned(post.postprocess(result));seconds=time.perf_counter()-started
            gold=record['review']['data']
            cases.append({'id':doc['id'],'production':score(gold,record['postprocess']),
                          'old_layout_new_clean':score(gold,cleaned(record['postprocess'])),
                          'new_pipeline':score(gold,new),'new_seconds_without_ocr':seconds,
                          'prediction':new})
            print(len(cases),len(docs),flush=True)
    summary={'split':a.split,'documents':len(cases),'scope':'only fully OCR-aligned business-review records; not full production accuracy','metrics':{}}
    for mode in ('production','old_layout_new_clean','new_pipeline'):
        summary['metrics'][mode]={field:sum(c[mode][field] for c in cases) for field in ('track','recipient','skus','items')}
        summary['metrics'][mode]['all_fields']=sum(all(c[mode].values()) for c in cases)
    (a.output/'cases.json').write_text(json.dumps(cases,ensure_ascii=False))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
