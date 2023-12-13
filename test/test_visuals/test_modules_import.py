import unittest
from importlib import import_module
from pathlib import Path


# class TestImports(unittest.TestCase):
#     """Can the various modules of pyciemss be successfully imported?"""

#     def test_imports(self):
#         # Collect modules.  Can be diretories (those with __init__.py)
#         # or files (things that end in .py but are not special names)

#         def as_import_target(path, root):
#             basis = path.relative_to(root.parent)
#             if basis.name == "__init__.py":
#                 return ".".join(basis.parent.parts)

#             if any(part.startswith(".") for part in basis.parts):
#                 return False

#             return ".".join(basis.with_suffix("").parts)

#         def try_import(import_name):
#             try:
#                 r = import_module(import_name)
#             except ModuleNotFoundError:
#                 return False
#             except ImportError:
#                 return False

#             return r.__name__ == import_name

#         root = Path(__file__).parent.parent / "src" / "pyciemss"

#         candidates = root.glob("**/*.py")
#         targets = [
#             as_import_target(n, root) for n in candidates if as_import_target(n, root)
#         ]
#         self.assertGreater(len(targets), 0, "No import targets found.")

#         results = {target: try_import(target) for target in targets}
#         if not all(results.values()):
#             failures = [m for m, r in results.items() if not r]
#             sep = "\n\t"
#             print(f"Failed imports:\n\t {sep.join([str(e) for e in failures])}")
#             self.assertEqual([], failures, "Imports failed")
