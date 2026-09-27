name: Minimal HMR2 Diagnostic

on:
  workflow_dispatch:

jobs:
  diagnostic:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android || true
          df -h /

      - name: Initial memory
        run: free -h

      - name: Install PyTorch CPU only
        run: |
          pip install --upgrade pip
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

      - name: Download checkpoint
        run: |
          mkdir -p ckpt
          cd ckpt
          wget -q -O checkpoint.ckpt \
            "https://huggingface.co/spaces/brjathu/HMR2.0/resolve/main/logs/train/multiruns/hmr2/0/checkpoints/epoch%3D35-step%3D1000000.ckpt"
          ls -lh

      - name: Memory before checkpoint load
        run: free -h

      - name: Test checkpoint load
        run: |
          python -u -c
          import os, torch, gc, subprocess
          
          print('=== PyTorch version:', torch.__version__)
          print('=== Before load ===')
          subprocess.run(['free', '-h'])
          
          ckpt_path = 'ckpt/checkpoint.ckpt'
          print(f'Loading: {ckpt_path}')
          
          # Patch weights_only
          _o = torch.load
          torch.load = lambda *a, **k: _o(*a, **{**k, 'weights_only': False})
          
          try:
              data = torch.load(ckpt_path, map_location='cpu')
              print('✓ Checkpoint loaded successfully')
              print(f'  Keys: {list(data.keys())[:10]}')
              
              if 'state_dict' in data:
                  n_params = len(data['state_dict'])
                  print(f'  state_dict entries: {n_params}')
              
              print('=== After load ===')
              subprocess.run(['free', '-h'])
              
              # Try to free
              del data
              gc.collect()
              print('=== After cleanup ===')
              subprocess.run(['free', '-h'])
              
          except Exception as e:
              print(f'❌ Failed to load: {type(e).__name__}: {e}')
              print('=== After failure ===')
              subprocess.run(['free', '-h'])
              raise
          "

      - name: Peak memory summary
        if: always()
        run: |
          echo "=== Final memory ==="
          free -h
