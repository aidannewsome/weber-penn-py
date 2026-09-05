# weber-penn-py

Weber and Penn's tree model, "Creation and Rendering of Realistic Trees" (SIGGRAPH 1995),
in one Python file with no dependencies. It grows the same tree from the same seed as
[weber-penn-net](https://github.com/aidannewsome/weber-penn-net), the C# port checked
against Arbaro, and `test_weberpenn.py` holds it to that.

```python
import json
from weberpenn import Parameters, Tree, Mesh

presets = json.load(open("presets.json"))
tree = Tree.grow(Parameters.from_paper(presets["Quaking Aspen"]), seed=13)
mesh = Mesh.of(tree, sides=8)
```

`tree.stems` are the trunk and every branch, each a run of sections with a frame and a
radius, its children, clones and leaves; `Mesh.of` is a plain tube-and-quad mesh with
texture coordinates in metres along the bark. `presets.json` is the paper's appendix
under the paper's own parameter names. Z is up, metres. MIT.
