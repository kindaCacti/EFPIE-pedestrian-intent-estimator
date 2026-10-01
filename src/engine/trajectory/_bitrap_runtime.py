"""Isolated loader for optional upstream BiTraP, without its old training stack.

Compatibility corrections are applied to parsed code in memory, leaving the
upstream checkout intact. Imports use a private namespace, never ``bitrap``.
"""

import ast
import hashlib
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType

_LOCK = RLock()


def source_digest(root):
    package = Path(root) / "bitrap"
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _Compatibility(ast.NodeTransformer):
    def __init__(self, namespace, module, device):
        self.namespace, self.module, self.device = namespace, module, str(device)

    def visit_ImportFrom(self, node):
        if node.module and (node.module == "bitrap" or node.module.startswith("bitrap.")):
            node.module = node.module.replace("bitrap", self.namespace, 1)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda" and not node.args:
            return ast.copy_location(ast.Call(func=ast.Attribute(node.func.value, "to", ast.Load()),
                                             args=[ast.Constant(self.device)], keywords=[]), node)
        for keyword in node.keywords:
            if keyword.arg == "device" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "cuda":
                keyword.value = ast.Constant(self.device)
            if keyword.arg == "batch_shape" and self.module in {"modeling.gmm2d", "modeling.gmm4d"}:
                keyword.value = ast.parse("log_pis.shape[:-1]", mode="eval").body
            if keyword.arg == "event_shape" and self.module in {"modeling.gmm2d", "modeling.gmm4d"}:
                dimension = 4 if self.module.endswith("4d") else 2
                keyword.value = ast.Tuple([ast.Constant(dimension)], ast.Load())
        if self.module in {"modeling.gmm2d", "modeling.gmm4d"} and any(k.arg == "batch_shape" for k in node.keywords):
            node.keywords.append(ast.keyword(arg="validate_args", value=ast.Constant(False)))
        if self.module == "modeling.bitrap_gmm":
            # Upstream ego-centric GMM retains ETH's 6-D future encoder and
            # 2-D goal reshapes; correct these to its configured box dimensions.
            if isinstance(node.func, ast.Attribute) and node.func.attr == "Linear" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 6:
                node.args[0] = ast.parse("self.cfg.GLOBAL_INPUT_DIM", mode="eval").body
            if isinstance(node.func, ast.Attribute) and node.func.attr == "reshape" and len(node.args) == 2 and isinstance(node.args[1], ast.List):
                shape = node.args[1]
                if len(shape.elts) == 3 and ast.unparse(shape.elts[0]) == "-1" and ast.unparse(shape.elts[1]) == "1":
                    expression = ast.unparse(node.args[0])
                    if expression.endswith((".mus", ".log_sigmas")):
                        shape.elts[2] = ast.parse("self.cfg.DEC_OUTPUT_DIM", mode="eval").body
                    elif expression.endswith(".corrs"):
                        shape.elts[2] = ast.parse("self.cfg.DEC_OUTPUT_DIM // 2", mode="eval").body
        return node


def load_bitrap(root, device):
    root = Path(root).resolve()
    package = root / "bitrap"
    if not (package / "modeling" / "bitrap_np.py").is_file():
        raise ImportError(f"Optional BiTraP source missing at {root}. Run bash tools/pull_bitrap.sh or set bitrap.upstream_root to a checkout of https://github.com/umautobots/bidirection-trajectory-predicter")
    namespace = "_sova_bitrap_" + hashlib.sha256((str(root) + str(device) + source_digest(root)).encode()).hexdigest()[:12]
    with _LOCK:
        if namespace + ".modeling.bitrap_gmm" not in sys.modules:
            for suffix in ("", ".modeling", ".modeling.dynamics", ".layers"):
                module = ModuleType(namespace + suffix)
                module.__path__ = [str(package / suffix.lstrip(".").replace(".", "/"))]
                module.__package__ = namespace + suffix
                sys.modules[module.__name__] = module
            modules = ("modeling.gmm2d", "modeling.gmm4d", "modeling.latent_net", "modeling.dynamics.utils",
                       "modeling.dynamics.integrator", "layers.loss", "modeling.bitrap_np", "modeling.bitrap_gmm")
            try:
                for name in modules:
                    path = package / (name.replace(".", "/") + ".py")
                    tree = _Compatibility(namespace, name, device).visit(ast.parse(path.read_text(), filename=str(path)))
                    ast.fix_missing_locations(tree)
                    module = ModuleType(namespace + "." + name)
                    module.__package__ = module.__name__.rsplit(".", 1)[0]
                    module.__file__ = str(path)
                    sys.modules[module.__name__] = module
                    exec(compile(tree, str(path), "exec"), module.__dict__)
            except BaseException:
                for key in list(sys.modules):
                    if key == namespace or key.startswith(namespace + "."):
                        del sys.modules[key]
                raise
        return (sys.modules[namespace + ".modeling.bitrap_np"].BiTraPNP,
                sys.modules[namespace + ".modeling.bitrap_gmm"].BiTraPGMM)
