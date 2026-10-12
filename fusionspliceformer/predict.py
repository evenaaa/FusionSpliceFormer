"""Released-model inference with event alignment and ensemble prediction."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model import FusionSpliceFormer
from .data.cache import PredictionDataset
from .io import sha256
from .console import Activity, EventProgress
from .runtime import (Progress, ScoreStore, output_lock, run_signature, open_run,
                      cleanup_run, cache_identity, publish_tsv)


def default_checkpoints():
    """Load the fixed official ten-model ensemble, never a silent subset."""
    from .released_model import released_checkpoints
    return released_checkpoints()


def prediction_main(device_name, argv=None):
    """CPU/GPU prediction entry point: trained weights and real inputs required."""
    import argparse
    import time
    parser=argparse.ArgumentParser(description='Predict splice pathogenicity from a VCF or variant/site TSV file.')
    parser.add_argument('-M','--model',dest='checkpoints',metavar='MODEL',nargs='+',type=Path,
                        help='Custom model override; default: official ten-model ensemble in weights/')
    parser.add_argument('-I','--input',dest='cache_root',metavar='INPUT',type=Path,required=True,
                        help='VCF/VCF.gz (automatic target mapping), or event TSV/TSV.gz with specified targets')
    parser.add_argument('-O','--output',metavar='OUTPUT',type=Path,required=True,help='New output TSV path')
    parser.add_argument('-B','--batch-size',metavar='N',type=int,default=4,help='Batch size (default: 4)')
    parser.add_argument('-G','--genome',default='hg38',choices=('hg38','GRCh38'),
                        help='Input genome build: hg38/GRCh38 only; convert other builds before prediction')
    parser.add_argument('-R','--reference-root',type=Path,default=Path('refs'),help='Reference resources (default: refs)')
    parser.add_argument('-P','--preprocessor-root',type=Path,default=Path('weights/preprocessors'),
                        help='Matching fitted numerical preprocessors (default: weights/preprocessors)')
    parser.add_argument('-W','--work-dir',type=Path,help='Temporary workspace (default: work/ beside the output file)')
    parser.add_argument('--keep-cache',action='store_true',help='Keep this run directory after successful prediction')
    parser.set_defaults(amp=False)
    if device_name == 'cuda':
        parser.add_argument('--amp',action='store_true',help='Use CUDA mixed precision (scores can differ from FP32)')
    parser.add_argument('--verbose',action='store_true',help='Show detailed preparation diagnostics')
    parser.add_argument('--details',action='store_true',help='Report all target-site events with input IDs, distances and scoring status (VCF default: highest-scoring site per variant)')
    parser.add_argument('--chunk-size',type=int,default=1024,metavar='N',
                        help='Maximum VCF events prepared at once (default: 1024); separate from model batch size')
    args=parser.parse_args(argv)
    if args.work_dir is None:
        args.work_dir = args.output.resolve().parent/'work'
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    if args.output.exists():
        parser.error(f'Output already exists; choose a new path: {args.output}')
    if not args.cache_root.exists():
        parser.error(f'Input not found: {args.cache_root}. Provide a VCF or target-site TSV; see README.md (Input).')
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use a compatible CUDA PyTorch build or predict_cpu.py.')
    if args.checkpoints is None:
        try:
            args.checkpoints = default_checkpoints()
            if args.cache_root.is_file():
                from .released_model import validate_preprocessors
                validate_preprocessors(args.preprocessor_root)
        except (ValueError, KeyError) as error:
            parser.error(str(error))
    for path in args.checkpoints:
        if not path.is_file():
            parser.error(f'Model not found: {path}. Restore the included weights/ directory or check your custom -M path.')
    started=time.perf_counter()
    torch.set_num_threads(2)
    if device_name == 'cuda' and not args.amp:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    try:
        with output_lock(args.output):
            if args.output.exists():
                raise ValueError(f'Output already exists; choose a new path: {args.output}')
            if args.verbose: print('Checking input, model and resource identities...',flush=True)
            signature=run_signature(args,device_name)
            work=open_run(args.work_dir,signature,verbose=args.verbose)
            try:
                _run_prediction(args,device_name,work,started)
            except BaseException:
                print(f'Run interrupted or failed. Saved progress: {work}\n'
                      'Rerun the same command to resume completed stages and saved predictions.',flush=True)
                raise
            if args.keep_cache:
                print(f'Cache retained: {work}',flush=True)
            else:
                try:
                    cleanup_run(work,args.work_dir,signature)
                    if args.verbose: print('Temporary cache and prediction progress removed.',flush=True)
                except OSError as error:
                    print(f'Prediction succeeded, but cache cleanup failed: {error}\nCache: {work}',flush=True)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


def _run_prediction(args,device_name,work,started):
    import time
    from .output import select_output
    cache_root=args.cache_root
    if cache_root.is_file():
        from .prepare import prepare_events, reference_paths
        from .input import read_events
        from .vcf import is_vcf
        paths = reference_paths(args.reference_root)
        if not paths.fasta.is_file():
            raise ValueError(f'GRCh38 reference FASTA not found under {args.reference_root}; see README.md (Prepare references)')
        if is_vcf(cache_root):
            from .streaming import run_vcf
            count = run_vcf(args, device_name, work, paths)
            print(f'Done | {count:,} result rows | total {time.perf_counter()-started:.1f}s | {args.output}',flush=True)
            return
        frame = read_events(cache_root)
        for region in frame.site_type.unique():
            pp = args.preprocessor_root/region/'preprocessor.json'
            if not pp.is_file():
                raise ValueError(f'Matching numerical preprocessor not found: {pp}')
        print(f'FusionSpliceFormer | {"GPU" if device_name=="cuda" else "CPU"} | '
              f'{len(args.checkpoints)} models | {len(frame):,} events',flush=True)
        with Activity('Preparing inputs',enabled=not args.verbose):
            cache_root = prepare_events(cache_root,work,args.reference_root,
                                        preprocessor_root=args.preprocessor_root,resume=True,progress=args.verbose)
            cache_identity(cache_root)
    else:
        print(f'FusionSpliceFormer | {"GPU" if device_name=="cuda" else "CPU"} | '
              f'{len(args.checkpoints)} models | prepared input',flush=True)
    def report_transform(report):
        return select_output(report, details=args.details)
    result=predict_ensemble(args.checkpoints,cache_root,args.output,args.batch_size,device_name,
                            mixed_precision=args.amp,state_dir=work/'scores',progress=True,verbose=args.verbose,
                            report_transform=report_transform)
    print(f'Done | {len(result):,} result rows | total {time.perf_counter()-started:.1f}s | {args.output}',flush=True)


def load_model(checkpoint, device='cpu'):
    # Load only checkpoints from trusted sources; unrestricted pickle is not used.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    from .released_model import tensor_identity
    if set(saved) != {'format_version','model_id','model_config','model_state_dict'} or saved.get('format_version')!='fusionspliceformer_inference_v1':
        raise ValueError('Use a released inference checkpoint; training checkpoints are not supported')
    if saved['model_id'] != tensor_identity(saved['model_state_dict']):
        raise ValueError('Checkpoint tensor identity mismatch')
    cfg=saved['model_config']
    model=FusionSpliceFormer.from_model_config(cfg)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    return model.to(device).eval(), saved




def format_predictions(result, cache_root):
    """Public event report with a single ensemble score and a model-predicted class."""
    from .output import OUTPUT_COLUMNS
    parts=[]
    for region in ('acceptor','donor'):
        path=Path(cache_root)/region/'metadata.tsv.gz'
        if path.is_file():
            parts.append(pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False))
    metadata=pd.concat(parts,ignore_index=True).set_index('event_key')
    if not metadata.index.is_unique or set(metadata.index)!=set(result.index):
        raise ValueError('Output metadata and prediction events differ')
    # Raw-input runs retain input row order across acceptor/donor batches.
    source=Path(cache_root).parent.parent/Path(cache_root).name/'events.tsv.gz'
    if source.is_file():
        order=pd.read_csv(source,sep='\t',usecols=['event_key'],dtype=str).event_key.tolist()
        if len(order)!=len(result) or set(order)!=set(result.index):
            raise ValueError('Input order and prediction events differ')
        result=result.loc[order]
    metadata=metadata.loc[result.index]
    if not metadata.site_type.equals(result.site_type):
        raise ValueError('Output metadata site types differ')
    rows=[]
    for key,row in metadata.iterrows():
        chrom,pos,ref,alt=row.variant_key.split(':')
        site_chrom,site_pos,strand,site_type=row.physical_site_id.split(':')
        if chrom!=site_chrom or site_type!=row.site_type or strand not in ('+','-'):
            raise ValueError(f'Inconsistent variant/site metadata: {key}')
        score=float(result.loc[key,'FusionSpliceFormer'])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Invalid score: {key}')
        rows.append({'CHROM':chrom,'POS':int(pos),'ID':row.get('input_id','.') or '.',
                     'REF':ref,'ALT':alt,'GENE':row.get('gene','.') or '.',
                     'TRANSCRIPTS':row.get('transcripts','.') or '.',
                     'SITE_POS':int(site_pos),'STRAND':strand,'SITE_TYPE':site_type,
                     'DISTANCE_BP':abs(int(pos)-int(site_pos)),'SCORE':score,
                     'PREDICTION':'Pathogenic' if score>=0.5 else 'Benign','STATUS':'OK'})
    return pd.DataFrame(rows,columns=OUTPUT_COLUMNS)


def display_score(score):
    """Readable scores without rounding a negative prediction across its threshold."""
    text=f'{score:.6f}'
    if (float(text)>=0.5)!=(score>=0.5) or (score>0 and float(text)==0):
        return repr(float(score))
    return text


def load_ensemble(checkpoints, device, verbose=False):
    """Validate and load the models once for a complete prediction run."""
    configs, identities, digests, models = [], [], [], []
    reporter = Progress('Checking model weights', len(checkpoints)) if verbose else None
    for path in checkpoints:
        model, saved = load_model(path, device)
        models.append(model)
        configs.append(saved['model_config'])
        identities.append(saved['model_id'])
        digests.append(sha256(path))
        if reporter: reporter.update(len(models), force=True)
    if not models or any(c != configs[0] for c in configs) or len(set(digests)) != len(digests):
        raise ValueError('Mixed model configurations or duplicate weights')
    if len(set(identities)) != len(identities):
        raise ValueError('Ensemble requires distinct model parameter identities')
    return models, digests


def predict_ensemble(checkpoints, cache_root, output, batch_size=4, device=None, *, mixed_precision=True,
                     state_dir=None, progress=False, verbose=False, report_transform=None,
                     model_bundle=None, publish=True, on_batch=None):
    """Finish all models for each region-specific batch before counting events."""
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    if batch_size < 1: raise ValueError('batch_size must be positive')
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoints=[Path(p).resolve() for p in checkpoints]
    if not checkpoints or len(checkpoints)!=len(set(checkpoints)):
        raise ValueError('Provide distinct checkpoints')
    cache_root=Path(cache_root)
    datasets,metadata,values,stores={},{},{},{}
    for region in ('acceptor','donor'):
        folder=cache_root/region
        if not folder.exists(): continue
        datasets[region]=PredictionDataset(folder)
        metadata[region]=pd.read_csv(folder/'metadata.tsv.gz',sep='\t',dtype=str)
        if len(metadata[region]) != len(datasets[region]):
            raise ValueError(f'Metadata/cache row counts differ: {folder}')
    if not metadata: raise ValueError('No acceptor/donor caches found')
    combined=pd.concat(metadata.values(),ignore_index=True).set_index('event_key')
    if combined.empty or not combined.index.is_unique: raise ValueError('Empty or duplicate prediction events')
    source=cache_root.parent.parent/cache_root.name/'events.tsv.gz'
    if source.is_file():
        order=pd.read_csv(source,sep='\t',usecols=['event_key'],dtype=str).event_key.tolist()
        if len(order)!=len(combined) or set(order)!=set(combined.index):
            raise ValueError('Input order and prediction events differ')
        combined=combined.loc[order]
    # Require the same architecture and distinct parameter sets.
    models, digests = model_bundle if model_bundle is not None else load_ensemble(checkpoints,device,verbose)
    if state_dir:
        import json
        from .io import atomic_json
        state_dir=Path(state_dir)
        identity={'layout':'event_ensemble_v1','weights':digests,'cache':cache_identity(cache_root),'batch_size':batch_size,
                  'device':str(device),'mixed_precision':mixed_precision,
                  'code':{str(p.relative_to(Path(__file__).parent)):sha256(p)
                          for p in sorted(Path(__file__).parent.rglob('*.py'))}}
        receipt=state_dir/'prediction.json'
        if receipt.exists():
            if json.loads(receipt.read_text(encoding='utf-8')) != identity:
                raise ValueError('Saved predictions belong to different inputs or settings')
        else:
            atomic_json(receipt,identity)
    initial={}
    jobs=[]
    ranks={key:i for i,key in enumerate(combined.index)}
    for region,dataset in datasets.items():
        store=ScoreStore(Path(state_dir)/'ensemble'/region,len(dataset)) if state_dir else None
        stores[region]=store
        values[region]=list(store.values) if store else []
        completed=len(values[region])
        if completed != len(dataset) and completed % batch_size:
            raise ValueError('Saved progress is not on a batch boundary')
        keys=metadata[region].event_key.tolist()
        initial.update(zip(keys[:completed],values[region]))
        for start in range(completed,len(dataset),batch_size):
            end=min(start+batch_size,len(dataset))
            jobs.append((min(ranks[k] for k in keys[start:end]),region,start,end))
    jobs.sort()
    reporter=EventProgress(combined,initial,display_score) if progress else None
    if on_batch: on_batch(sum(len(v) for v in values.values()))
    try:
        with torch.inference_mode():
            for _,region,start,end in jobs:
                batch=torch.utils.data.default_collate([datasets[region][i] for i in range(start,end)])
                inputs={k:v.to(device) for k,v in batch.items()}
                probabilities=[]
                for model in models:
                    with torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda' and mixed_precision):
                        out=model(region=region,**inputs)
                    scores=out.logits.softmax(-1)[:,1].float().cpu().tolist()
                    if len(scores)!=end-start or not np.isfinite(scores).all() or not np.all((np.asarray(scores)>=0)&(np.asarray(scores)<=1)):
                        raise ValueError('Invalid model probabilities')
                    probabilities.append(scores)
                # Same model order and float64 mean as the model-first implementation.
                ensemble=np.stack(probabilities).mean(axis=0).tolist()
                if stores[region]: stores[region].save(ensemble)
                values[region].extend(ensemble)
                if on_batch: on_batch(sum(len(v) for v in values.values()))
                if reporter: reporter.advance(metadata[region].event_key.iloc[start:end].tolist(),ensemble)
        if reporter and not jobs: reporter.advance([],[])
    finally:
        if reporter: reporter.close()
    result=pd.concat([metadata[r][['event_key','site_type']].assign(FusionSpliceFormer=values[r])
                      for r in datasets],ignore_index=True).set_index('event_key')
    if verbose: print('Writing the final TSV...',flush=True)
    report=format_predictions(result,cache_root)
    for dataset in datasets.values():
        dataset.close()
    if report_transform is not None:
        report=report_transform(report)
    if not publish:
        return report
    if state_dir:
        publish_tsv(report,output,Path(state_dir).parent,display_score)
    else:
        import tempfile
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.fusion_publish_',dir=output.parent) as folder:
            publish_tsv(report,output,folder,display_score)
    return report
