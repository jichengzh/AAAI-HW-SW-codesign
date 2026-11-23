"""Diagnostic CUDA proxy: replay device input to host before compute."""
import ctypes,hashlib,json,os
from pathlib import Path

class ReadbackCuda:
 def __init__(self,base,path):self.base=base;self.path=Path(path);self.index=0
 def __getattr__(self,name):return getattr(self.base,name)
 def copy(self,dst,src,size,kind,stream):
  if kind!=1:return self.base.copy(dst,src,size,kind,stream)
  self.path.mkdir(parents=True,exist_ok=True);index=self.index;self.index+=1
  expected=ctypes.string_at(src,size)
  row={'index':index,'size':size,'expected_sha256':hashlib.sha256(expected).hexdigest(),'stage':'begin'}
  self._row(row)
  self.base.copy(dst,src,size,kind,stream)
  buffer=ctypes.create_string_buffer(size)
  self.base.copy(ctypes.addressof(buffer),dst,size,2,stream)
  self.base.sync(stream)
  raw=buffer.raw
  p=self.path/('input_%05d.bin'%index)
  with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  row={'index':index,'size':size,'expected_sha256':hashlib.sha256(expected).hexdigest(),
       'sha256':hashlib.sha256(raw).hexdigest(),'match':raw==expected,'stage':'complete'}
  self._row(row)
  if not row['match']:raise ValueError('H2D input readback differs')
 def _row(self,row):
  with (self.path/'rows.jsonl').open('a') as f:
   f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
