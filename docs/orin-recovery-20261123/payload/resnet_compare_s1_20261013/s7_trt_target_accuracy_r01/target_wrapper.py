"""Preserve binary stdout and report the actual bounded target process exit."""
import subprocess,sys

def run(command):
 code=127
 try:
  code=subprocess.run(command,check=False).returncode
 finally:
  print('REMOTE_FULLVAL_EXIT='+str(code),file=sys.stderr,flush=True)
 return code if code>=0 else 128-code
if __name__=='__main__':raise SystemExit(run(sys.argv[1:]))
