import ast, json, re, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', result['text'], re.S)
assert len(blocks) == 1, 'Expected one complete Python block'
code = blocks[0]
tree = ast.parse(code)
assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == 'fibonacci'
for node in ast.walk(tree):
    assert not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)), 'Unexpected code construct'
    if isinstance(node, ast.Attribute): assert node.attr in ('append', 'extend'), 'Unexpected attribute'
    if isinstance(node, ast.Name): assert not node.id.startswith('__'), 'Unexpected private name'
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        assert node.func.id in ('isinstance','type','range','len','list','int','bool','TypeError','ValueError'), 'Unexpected function call'
namespace = {'__builtins__': {'isinstance':isinstance, 'type':type, 'range':range, 'len':len,
                            'list':list, 'int':int, 'bool':bool, 'TypeError':TypeError, 'ValueError':ValueError}}
exec(compile(tree,'generated_fibonacci.py','exec'), namespace)
f = namespace['fibonacci']
checks = []
for n in (0,1,2,5,10,200):
    expected=[]; a,b=0,1
    for _ in range(n): expected.append(a); a,b=b,a+b
    actual=f(n); assert isinstance(actual,list) and actual==expected, f'Incorrect sequence for {n}'
    checks.append({'case':n,'passed':True})
for value, error in [(-1,ValueError),(-200,ValueError),(True,TypeError),(False,TypeError),('5',TypeError),(2.5,TypeError),(None,TypeError)]:
    try: f(value)
    except error: checks.append({'case':repr(value),'passed':True})
    else: raise AssertionError(f'Did not reject {value!r}')
assert result['finish_reason'] == 'stop', 'Response did not finish naturally'
print(json.dumps({'passed':True,'checks':checks}))
