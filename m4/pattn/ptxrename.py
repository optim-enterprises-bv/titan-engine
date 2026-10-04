"""ptxrename.py INSTANCES.rs IN.ptx OUT.ptx: give the short oxide decode entries (`bd<n>`) back their reference (mangled)
names, so ptxkind.py compares them with a PTX built before the rename."""
import re, sys
ox = dict((o, n) for n, o in re.findall(r'name: "([^"]+)", ox: "(bd\d+)"', open(sys.argv[1]).read()))
s = open(sys.argv[2]).read()
s = re.sub(r'\b(bd\d+)(?=(_param_\d+)?\b)', lambda m: ox.get(m.group(1), m.group(1)), s)
open(sys.argv[3], 'w').write(s)
print(f"ptxrename: {len(ox)} short names mapped")
