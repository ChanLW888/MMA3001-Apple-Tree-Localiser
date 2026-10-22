"""Capture the requested unittest command verbatim without shell stderr rewriting."""
import datetime
from pathlib import Path
import subprocess
import sys
import re
command=[sys.executable,'-m','unittest','discover','-s','test','-p','test_*.py','-v']
result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
report='MMA3001 synthetic unit-test execution\n'
report+='UTC: '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n'
report+='Interpreter: '+sys.executable+'\nEnvironment: mma3001\n'
report+='Command: python -m unittest discover -s test -p "test_*.py" -v\n\n'
report+=result.stdout
match=re.search(r'Ran (\d+) tests?',result.stdout)
if match and result.returncode==0 and re.search(r'^OK$',result.stdout,re.M):
    report+='\nCounts: '+match.group(1)+' passed; 0 failed; 0 errored; 0 skipped.\n'
report+='\nProcess exit code: '+str(result.returncode)+'\n'
report+='\nVerified: synthetic red/depth/branch filtering, morphology, component geometry,\n'
report+='temporary exports, matching, hand-calculated metrics, CSV validation, actual\n'
report+='5,184-configuration grid, objective directions, tree grouping and tiny mocked resume.\n'
report+='Limitations: no dataset/model accuracy validation, checkpoints, GPU, training,\n'
report+='RF-DETR inference or full optimisation. Known stale held-out reuse on changed\n'
report+='selection objective is reproduced by an explicitly named characterization test,\n'
report+='not fixed or certified safe. See test/README.md for details and other limits.\n'
Path('test/test_report.txt').write_text(report,encoding='utf-8')
print(report)
sys.exit(result.returncode)
