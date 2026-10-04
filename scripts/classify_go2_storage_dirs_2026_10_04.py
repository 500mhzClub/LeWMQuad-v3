"""Storage review, 4 October 2026: size directories by file type. Read-only, filename and size only.

Walks each listed directory without following links, never entering or counting `sealed*` names, and reports
allocated GiB, file count, newest modification date and the six largest extensions. Hard-linked files are
counted once per path here; measure shared trees with `du`, which counts each inode once.

Usage: classify_go2_storage_dirs_2026_10_04.py LIST_OF_DIRS OUT.json
"""
import os, sys, json, time, collections
from concurrent.futures import ThreadPoolExecutor
def cls(d):
    by=collections.Counter(); n=0; newest=0.
    for dp, dirs, files in os.walk(d, followlinks=False):
        dirs[:]=[x for x in dirs if 'sealed' not in x]
        for f in files:
            if 'sealed' in f: continue
            try: st=os.lstat(os.path.join(dp,f))
            except OSError: continue
            ext=os.path.splitext(f)[1].lower() or '(none)'
            by[ext]+=st.st_blocks*512; n+=1; newest=max(newest,st.st_mtime)
    return dict(path=d, gib=round(sum(by.values())/2**30,2), files=n, newest=time.strftime('%Y-%m-%d',time.localtime(newest)) if newest else None, ext={k:round(v/2**30,2) for k,v in by.most_common(6)})
dirs=[l.strip() for l in open(sys.argv[1]) if l.strip()]
with ThreadPoolExecutor(12) as ex: res=list(ex.map(cls, dirs))
json.dump(res, open(sys.argv[2],'w'), indent=0)
for r in sorted(res, key=lambda r:-r['gib']): print(f"{r['gib']:8.2f} {r['newest']} {r['files']:>8} {r['path'].replace('/home/andrewknowles','~')[:95]} {r['ext']}")
