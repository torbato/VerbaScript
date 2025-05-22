# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['VerbaScript.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\giova\\Developer\\torbato\\VerbaScript\\venv\\Lib\\site-packages\\whisper\\assets\\mel_filters.npz', 'whisper/assets'), ('C:\\Users\\giova\\Developer\\torbato\\VerbaScript\\venv\\Lib\\site-packages\\whisper\\assets\\multilingual.tiktoken', 'whisper/assets'), ('assets/whisper_models/tiny.pt', 'assets/whisper_models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VerbaScript 0.7',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\I.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VerbaScript 0.7',
)
