from pathlib import Path
import subprocess
import sys

import visualkeras


def test_build_and_runtime_versions_match_without_importing_runtime_dependencies(tmp_path):
    setup = Path(__file__).resolve().parents[1] / "setup.py"
    script = """
import runpy
import sys

class BlockRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'visualkeras', 'tensorflow', 'keras', 'numpy', 'PIL', 'aggdraw'}:
            raise AssertionError('Build metadata imported runtime dependency: ' + fullname)

sys.meta_path.insert(0, BlockRuntimeImports())
sys.argv = [sys.argv[1], '--version']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(setup)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == visualkeras.__version__
