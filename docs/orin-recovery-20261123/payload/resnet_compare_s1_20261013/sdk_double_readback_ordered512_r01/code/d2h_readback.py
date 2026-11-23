"""Diagnostic proxy: compare two separately synchronized output readbacks."""
import ctypes,hashlib,json,os
from pathlib import Path
from h2d_readback import ReadbackCuda

class DoubleReadbackCuda(ReadbackCuda):
 def __init__(self,base,path):
  super().__init__(base,Path(path)/'h2d_readback');self.output_path=Path(path)/'d2h';self.output_index=0
 def copy(self,dst,src,size,kind,stream):
  if kind!=2:return super().copy(dst,src,size,kind,stream)
  self.output_path.mkdir(parents=True,exist_ok=True);i=self.output_index;self.output_index+=1
  self._output_row({'index':i,'stage':'begin','size':size})
  self.base.copy(dst,src,size,kind,stream);self.base.sync(stream)
  first=ctypes.string_at(dst,size);self._save('first',i,first)
  first_sha=hashlib.sha256(first).hexdigest()
  self._output_row({'index':i,'stage':'first_complete','size':size,'first_sha256':first_sha})
  other=ctypes.create_string_buffer(size)
  self.base.copy(ctypes.addressof(other),src,size,kind,stream);self.base.sync(stream)
  second=other.raw;self._save('second',i,second)
  row={'index':i,'stage':'complete','size':size,'first_sha256':first_sha,
       'second_sha256':hashlib.sha256(second).hexdigest(),'match':first==second}
  self._output_row(row)
  if not row['match']:raise ValueError('D2H independent readbacks differ')
 def _save(self,which,index,raw):
  with (self.output_path/('output_%s_%05d.bin'%(which,index))).open('xb') as f:
   f.write(raw);f.flush();os.fsync(f.fileno())
 def _output_row(self,row):
  with (self.output_path/'rows.jsonl').open('a') as f:
   f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
