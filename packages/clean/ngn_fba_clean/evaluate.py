"""Evaluate a selected checkpoint once on frozen CSVs; no fitting or threshold search."""
import argparse
import csv
import json
import time
from pathlib import Path
import torch
from train import KeepDropLightning, load_rows_from_csv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--splits',nargs='+',choices=['validation','test','llm_holdout'],default=['validation','test','llm_holdout'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    summaries={}
    for kind in ('track','sku','recipient'):
        run=args.run/kind
        checkpoints=list((run/'checkpoints').glob('best-*.ckpt'))
        assert len(checkpoints)==1,checkpoints
        vocab=json.loads((run/'vocab.json').read_text())
        model=KeepDropLightning.load_from_checkpoint(str(checkpoints[0]),vocab=vocab,map_location='cpu').to('cuda').eval()
        train_rows=load_rows_from_csv(str(args.data/kind/'train.csv'),'raw','clean','group',512)
        train_raws={r['_raw_norm'] for r in train_rows}
        summaries[kind]={'checkpoint':checkpoints[0].name}
        for split in args.splits:
            rows=load_rows_from_csv(str(args.data/kind/(split+'.csv')),'raw','clean','group',512)
            output=[]
            started=time.perf_counter()
            with torch.inference_mode():
                for start in range(0,len(rows),128):
                    batch=rows[start:start+128]
                    lengths=[len(r['_raw_norm']) for r in batch]
                    inputs=torch.zeros((len(batch),max(lengths)),dtype=torch.long,device='cuda')
                    for i,r in enumerate(batch):
                        inputs[i,:lengths[i]]=torch.tensor([vocab.get(c,vocab['<UNK>']) for c in r['_raw_norm']],device='cuda')
                    mask=model(inputs).sigmoid().ge(.5).cpu().tolist()
                    for r,keep in zip(batch,mask):
                        raw,clean=r['_raw_norm'],r['_clean_norm']
                        pred=''.join(c for c,k in zip(raw,keep) if k)
                        output.append({'file':r['file'],'raw':raw,'target':clean,'prediction':pred,'correct':pred==clean,'identity_correct':raw==clean,'unseen_raw':raw not in train_raws})
            summaries[kind][split]={'count':len(output),'exact':sum(r['correct'] for r in output),'identity_baseline_exact':sum(r['identity_correct'] for r in output),'unseen_raw_count':sum(r['unseen_raw'] for r in output),'unseen_raw_exact':sum(r['unseen_raw'] and r['correct'] for r in output),'batch_total_seconds':time.perf_counter()-started}
            (run/(split+'-predictions.json')).write_text(json.dumps(output,ensure_ascii=False))
        del model
        torch.cuda.empty_cache()
    (args.run/'evaluation.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
