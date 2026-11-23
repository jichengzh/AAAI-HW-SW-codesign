"""Fixed segmentation; one independent context per planned segment."""
PLAN=[(i*256,256) for i in range(9)]+[(2304,128)]

def validate_segment(start,count):
 if (start,count) not in PLAN:raise ValueError('segment outside fixed plan')
 return True
