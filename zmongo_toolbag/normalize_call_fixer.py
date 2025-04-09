import ast
import astor

# Load the contents of the zmagnum.py file
file_path = "/home/overlordx/PycharmProjects/zmongo_retriever/zmongo_toolbag/zmagnum.py"
with open(file_path, "r") as f:
    source_code = f.read()

# Parse the source code into an AST
tree = ast.parse(source_code)

# Visitor class to rewrite method calls to _normalize_collection_name
class NormalizeCallFixer(ast.NodeTransformer):
    def visit_Call(self, node):
        # Target method call: self._normalize_collection_name(...)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "_normalize_collection_name":
                # Ensure it's a method of self (i.e., self._normalize_collection_name(...))
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                    # Rewrite as: ZMagnum._normalize_collection_name(...)
                    node.func = ast.Attribute(
                        value=ast.Name(id="ZMagnum", ctx=ast.Load()),
                        attr="_normalize_collection_name",
                        ctx=ast.Load(),
                    )
        return self.generic_visit(node)

# Apply the transformation
fixer = NormalizeCallFixer()
fixed_tree = fixer.visit(tree)

# Convert AST back to source code
fixed_code = astor.to_source(fixed_tree)

# Write the fixed code back to file
with open(file_path, "w") as f:
    f.write(fixed_code)

fixed_code[:1000]  # Show a snippet to confirm it was applied correctly
