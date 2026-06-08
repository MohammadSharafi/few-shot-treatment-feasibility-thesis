import subprocess, sys
cmd=['xelatex','-file-line-error','-interaction=batchmode','thesis_fa.tex']
sys.exit(subprocess.run(cmd).returncode)
