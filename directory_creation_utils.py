models_dir = os.path.join(base_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created missing directory: {models_dir}")