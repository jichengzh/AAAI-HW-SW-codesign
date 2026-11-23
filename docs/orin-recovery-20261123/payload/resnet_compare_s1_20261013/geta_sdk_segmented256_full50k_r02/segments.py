"""Fixed segmentation; one independent context per planned segment."""
PLAN=[(start,min(256,50000-start)) for start in range(0,50000,256)]

def validate_segment(start,count):
 if (start,count) not in PLAN:raise ValueError('segment outside fixed plan')
 return True
