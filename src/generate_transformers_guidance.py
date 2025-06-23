#!/usr/bin/env python
"""Guidance JSON-schema generation – mirrors `generate_transformers_outlines`."""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Union
import torch, guidance
from guidance import models as gmodels
from tqdm import tqdm
# ------------------------------------------------------------------
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1"); os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")
if hasattr(torch, "compile"): torch.compile = lambda m,*a,**k: m  # type: ignore
BATCH_SIZE = 1; MAX_NEW_TOKENS = 2048; MAX_RECORDS: int | None = None
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------

def _sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        for k, v in list(x.items()):
            if k == "type" and isinstance(v, list):
                x[k] = next((i for i in v if i != "null"), v[0])
            elif k in {"if","then","else","oneOf","anyOf","allOf","not","propertyNames"} or k.startswith("$"):
                x.pop(k, None)
                continue
            _sanitize(v)
    elif isinstance(x, list):
        for i in x: _sanitize(i)
    return x

def _prep_schema(s: Union[str, Dict[str,Any]])->str:
    return json.dumps(_sanitize(json.loads(json.dumps(s)))) if isinstance(s,(dict,list)) else s

def _load_prompt(p: Path)->str:
    return p.read_text()

@lru_cache(maxsize=None)
def _prog(tpl: str, schema_json: str):
    txt = f"{{#system}}Return JSON only.{{/system}}\n{tpl}\n\n{{#assistant}}{{json name='answer' schema=_s}}{{/assistant}}"
    return guidance(txt)

def _process(llm, prompts: List[str], schemas: List[Any], recs: List[Dict[str,Any]], out_path: Path, task: str, mnt: int):
    out_path.mkdir(parents=True, exist_ok=True)
    out: List[Dict[str,Any]] = []
    with llm.session() as sess:
        for prm, sch, r in tqdm(list(zip(prompts, schemas, recs)), desc=task):
            s_json = _prep_schema(sch)
            res = _prog(prm, s_json)(llm=sess, _s=json.loads(s_json), max_tokens=mnt)
            out.append({**r, "generated_output": res["answer"].strip()})
    json.dump(out, open(out_path/"generated.json", "w"), indent=4)

# ------------------------------- task wrappers ---------------------------

def _gen_generic(llm, name: str, d: Path, o: Path, pt: str, mnt: int):
    bench = json.load(open(d/"bench.json"))[: (MAX_RECORDS or 10**9)]
    tpl = _load_prompt(d/f"prompt{'_'+pt if pt else ''}.txt")
    prompts=[ (tpl.replace("{input}", b["input"]) if "{input}" in tpl else tpl+f"\n\nInput: {b['input']}\nOutput:") for b in bench]
    schema_file=d/"schema.json"
    schemas=[json.load(schema_file.open())]*len(prompts) if schema_file.exists() else [b.get("json_schema", b.get("schema")) for b in bench]
    _process(llm, prompts, schemas, bench, o, name, mnt)

def _gen_wikibio(llm, d: Path, o: Path, pt: str, mnt: int):
    bench=json.load(open(d/"bench.json"))[: (MAX_RECORDS or 10**9)]
    tpl=_load_prompt(d/f"prompt{'_'+pt if pt else ''}.txt")
    prompts,schemas=[],[]
    for b in bench:
        prompts.append(tpl.replace("{{EXPECTED_KEYS}}", ", ".join(f"'{k}'" for k in b["keys"]))+f"\n\nInput: {b['input']}\nOutput:")
        schemas.append(b.get("json_schema", b.get("schema")))
    _process(llm, prompts, schemas, bench, o, "2-wiki_bio", mnt)

def _gen_apibank(llm, d: Path, o: Path, pt: str, mnt: int):
    bench=json.load(open(d/"bench.json"))[: (MAX_RECORDS or 10**9)]
    tpl=_load_prompt(d/f"prompt{'_'+pt if pt else ''}.txt")
    prompts=[tpl.replace("{input}", b["input"]).replace("{instruction}", b["instruction"]) for b in bench]
    schemas=[b.get("json_schema", b.get("schema")) for b in bench]
    _process(llm, prompts, schemas, bench, o, "5-api_bank", mnt)

# ------------------------------- main ------------------------------------

def main():
    global BATCH_SIZE, MAX_NEW_TOKENS, MAX_RECORDS
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--prompt-type", choices=["", "low"], default="")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE); ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--max-records", type=int); ap.add_argument("--start-from", choices=["1-rotowire","2-wiki_bio","3-few_nerd","4-TOPv1","5-api_bank","6-reasoning"])
    ap.add_argument("--no-flash-attn", dest="flash_attn", action="store_false"); ap.set_defaults(flash_attn=True)
    args=ap.parse_args(); BATCH_SIZE, MAX_NEW_TOKENS, MAX_RECORDS=args.batch_size, args.max_new_tokens, args.max_records
    # logging
    logs=Path("logs"); logs.mkdir(exist_ok=True)
    suff=f"_{args.prompt_type}" if args.prompt_type else ""; slug=args.model.split("/")[-1]
    logf=logs/f"generate_{slug}_transformers_guidance{suff}.log"; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(logf, mode="a"), logging.StreamHandler()], force=True)
    if logf.exists(): logger.info("RERUN")
    # model
    extra={"attn_implementation":"flash_attention_2"} if args.flash_attn else {}
    device="cuda" if torch.cuda.is_available() else "cpu"
    llm=gmodels.Transformers(args.model, device=device, dtype=(torch.bfloat16 if device=="cuda" else torch.float32), trust_remote_code=True, **extra)
    base=Path("data/clean"); res_root=Path("results")/f"{slug}_transformers_guidance{suff}"
    # tasks
    tasks={
        "1-rotowire": lambda: _gen_generic(llm,"1-rotowire",base/"1-rotowire",res_root/"1-rotowire",args.prompt_type,args.max_new_tokens),
        "2-wiki_bio": lambda: _gen_wikibio(llm,base/"2-wiki_bio",res_root/"2-wiki_bio",args.prompt_type,args.max_new_tokens),
        "3-few_nerd": lambda: _gen_generic(llm,"3-few_nerd",base/"3-few_nerd",res_root/"3-few_nerd",args.prompt_type,args.max_new_tokens),
        "4-TOPv1": lambda: _gen_generic(llm,"4-TOPv1",base/"4-TOPv1",res_root/"4-TOPv1",args.prompt_type,args.max_new_tokens),
        "5-api_bank": lambda: _gen_apibank(llm,base/"5-api_bank",res_root/"5-api_bank",args.prompt_type,args.max_new_tokens),
    }
    order=list(tasks.keys());
    if args.start_from and args.start_from!="6-reasoning": order=order[order.index(args.start_from):]
    for k in order:
        logger.info(f"Task {k}"); t0=time.perf_counter(); tasks[k](); logger.info(f"done in {time.perf_counter()-t0:.1f}s")
    if not args.start_from or args.start_from=="6-reasoning" or args.start_from not in tasks:
        for sub in ["GSM8K","last_letter"]:
            name=f"6-reasoning/{sub}"; logger.info(name); _gen_generic(llm,name,base/"6-reasoning"/sub,res_root/"6-reasoning"/sub,args.prompt_type,512)

if __name__=="__main__":
    main() 