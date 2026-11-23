"""Report the actual bounded target command exit on stderr, never binary stdout."""
import subprocess,sys
def run(command):
    result=subprocess.run(command,check=False)
    print('REMOTE_SDK_FULLVAL_EXIT='+str(result.returncode),file=sys.stderr,flush=True)
    return result.returncode
if __name__=='__main__':raise SystemExit(run(sys.argv[1:]))
