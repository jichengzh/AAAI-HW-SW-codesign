"""Select the isolated target ORT/NumPy tree before importing either."""
def configure(paths,modules,machine,version,root):
    if machine=='aarch64' and tuple(version[:2])==(3,12):
        if 'numpy' in modules or 'onnxruntime' in modules:raise ValueError('NumPy/ORT preloaded before isolated bootstrap')
        paths.insert(0,root)
