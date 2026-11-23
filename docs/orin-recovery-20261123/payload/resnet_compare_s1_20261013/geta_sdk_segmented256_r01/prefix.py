"""Bounded diagnostic prefix and historical input binding."""
COUNT=2432

def select_rows(rows):
 if len(rows)<COUNT:raise ValueError('insufficient dataset')
 return rows[:COUNT]

def verify_prior(actual,old,failed_sha):
 if len(actual)!=COUNT or len(old)!=2416:raise ValueError('identity count')
 if actual[:2416]!=old or actual[2429]!=failed_sha:raise ValueError('historical inputs differ')
 return True

def verify_pins(pins):
 import hashlib
 from pathlib import Path
 for path,expected in pins.items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('file hash mismatch '+path)
 return True
