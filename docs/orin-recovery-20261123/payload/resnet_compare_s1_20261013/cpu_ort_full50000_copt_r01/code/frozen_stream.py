"""Bounded authenticated transport; fixed batch-one target accuracy, not timing."""
import hashlib,json,struct,zlib
import numpy as np
CHUNK=16
ROW_BYTES=3*224*224*4
MAX_PAYLOAD=CHUNK*ROW_BYTES+65536

def digest(data): return hashlib.sha256(data).hexdigest()
def check(ok,message):
 if not ok: raise ValueError(message)
def exact(stream,size):
 blocks=[];left=size
 while left:
  chunk=stream.read(left)
  check(bool(chunk),'Truncated stream')
  blocks.append(chunk);left-=len(chunk)
 return b''.join(blocks)
def send(stream,header,payload=b''):
 raw=json.dumps(header,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 check(len(raw)<=65536 and len(payload)<=MAX_PAYLOAD,'Frame too large')
 stream.write(struct.pack('!II',len(raw),len(payload))+raw+payload);stream.flush()
def receive(stream):
 nh,npayload=struct.unpack('!II',exact(stream,8))
 check(0<nh<=65536 and npayload<=MAX_PAYLOAD,'Frame too large')
 return json.loads(exact(stream,nh)),exact(stream,npayload)
def encode_chunk(index,rows,inputs):
 check(inputs.dtype==np.float32 and inputs.shape==(len(rows),3,224,224),'Input dtype/shape')
 check(0<len(rows)<=CHUNK and np.isfinite(inputs).all(),'Input count/finite')
 raw=np.ascontiguousarray(inputs).tobytes();packed=zlib.compress(raw,1)
 header={'type':'chunk','index':index,'count':len(rows),'ids':[r['sample_id'] for r in rows],
         'row_sha256':[digest(x.tobytes()) for x in inputs],
         'raw_sha256':digest(raw),'payload_sha256':digest(packed)}
 return header,packed
def decode_chunk(header,payload,index,rows):
 n=header['count'];check(type(n) is int and 0<n<=CHUNK,'Chunk count')
 check(header['type']=='chunk' and header['index']==index,'Chunk order')
 check(header['ids']==[r['sample_id'] for r in rows[index:index+n]] and len(header['ids'])==n,'IDs/order')
 check(digest(payload)==header['payload_sha256'],'Payload SHA')
 dec=zlib.decompressobj();raw=dec.decompress(payload,n*ROW_BYTES+1)
 check(len(raw)==n*ROW_BYTES and dec.eof and not dec.unused_data and not dec.unconsumed_tail,'Decompression length/trailing data')
 check(digest(raw)==header['raw_sha256'],'Raw SHA')
 x=np.frombuffer(raw,dtype=np.float32).reshape(n,3,224,224)
 check(np.isfinite(x).all() and [digest(r.tobytes()) for r in x]==header['row_sha256'],'Row SHA/finite')
 return x
def advance(chain,header,payload):
 return digest(bytes.fromhex(chain)+json.dumps(header,sort_keys=True,separators=(',',':')).encode()+payload)
def serve(reader,writer,rows,infer,identity):
 send(writer,{'type':'ready','identity':identity})
 index=0;chain='00'*32
 while True:
  header,payload=receive(reader)
  if header.get('type')=='end':
   check(not payload and index==len(rows)==header['count'],'Incomplete final count')
   send(writer,{'type':'end','status':'COMPLETE','count':index,'chain_sha256':chain});return
  x=decode_chunk(header,payload,index,rows)
  output=[]
  for row in x:
   logits=np.asarray(infer(row[None]),dtype=np.float32)
   check(logits.shape==(1,1000) and np.isfinite(logits).all(),'Output shape/finite')
   output.append(logits)
  raw=np.concatenate(output).astype('<f4').tobytes()
  reply={'type':'logits','index':index,'count':len(x),'ids':header['ids'],
         'row_sha256':header['row_sha256'],'input_raw_sha256':header['raw_sha256'],
         'logits_sha256':digest(raw)}
  chain=advance(chain,reply,raw);send(writer,reply,raw);index+=len(x)
def verify_reply(header,payload,request):
 check(header.get('type')=='logits','Expected logits')
 for key in ('index','count','ids','row_sha256'):check(header[key]==request[key],'Response binding '+key)
 check(header['input_raw_sha256']==request['raw_sha256'] and digest(payload)==header['logits_sha256'],'Response SHA')
 check(len(payload)==request['count']*1000*4,'Response length')
 logits=np.frombuffer(payload,dtype='<f4').reshape(request['count'],1000)
 check(np.isfinite(logits).all(),'Nonfinite response');return logits

def evaluate(reader,writer,dataset,output,expected_identity):
 """Host GT remains external; no labels sent to target."""
 from pathlib import Path
 output=Path(output);output.mkdir(exist_ok=False)
 ready,payload=receive(reader)
 check(ready.get('type')=='ready' and not payload,'Target handshake')
 for k,v in expected_identity.items():check(ready['identity'].get(k)==v,'Target identity '+k)
 (output/'target_identity.json').write_text(json.dumps(ready['identity'],indent=2))
 count=top1=top5=0;chain='00'*32
 with (output/'predictions.jsonl').open('x') as record,(output/'logits.f32').open('xb') as rawout:
  for start in range(0,len(dataset),CHUNK):
   rows=dataset.rows[start:start+CHUNK];items=[dataset[i] for i in range(start,start+len(rows))]
   x=np.stack([np.asarray(item[0],dtype=np.float32) for item in items])
   check([int(v[1]) for v in items]==[r['target'] for r in rows],'External GT mismatch')
   request,packed=encode_chunk(start,rows,x);send(writer,request,packed)
   reply,raw=receive(reader);logits=verify_reply(reply,raw,request)
   chain=advance(chain,reply,raw);rawout.write(raw)
   # Match PyTorch topk semantics, including tie handling, from complete logits.
   import torch
   choices=torch.from_numpy(logits.copy()).topk(5,dim=1).indices.numpy()
   for offset,(row,pred) in enumerate(zip(rows,choices)):
    one=int(pred[0]==row['target']);five=int(row['target'] in pred)
    top1+=one;top5+=five;count+=1
    record.write(json.dumps({'index':start+offset,'sample_id':row['sample_id'],'target':row['target'],
      'input_sha256':request['row_sha256'][offset],'logits_sha256':digest(logits[offset].tobytes()),
      'top5':pred.tolist(),'top1_correct':one,'top5_correct':five,'chunk_chain_sha256':chain})+'\n')
   record.flush();rawout.flush()
   if count%1600==0:print(json.dumps({'samples':count}),flush=True)
  send(writer,{'type':'end','count':count})
  final,payload=receive(reader)
  check(not payload and final=={'type':'end','status':'COMPLETE','count':len(dataset),'chain_sha256':chain},'Final receipt')
 return {'samples':count,'top1_correct':top1,'top5_correct':top5,'top1_pp':100*top1/count,
         'top5_pp':100*top5/count,'chain_sha256':chain,'target_identity':ready['identity']}
