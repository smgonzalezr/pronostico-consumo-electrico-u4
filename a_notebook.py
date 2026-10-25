import nbformat, re, sys
src = open(sys.argv[1]).read()
celdas = []
for bloque in re.split(r'^# %%', src, flags=re.M)[1:]:
    if bloque.startswith(' [markdown]'):
        lineas = bloque.split('\n')[1:]
        txt = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in lineas).strip()
        celdas.append(nbformat.v4.new_markdown_cell(txt))
    else:
        celdas.append(nbformat.v4.new_code_cell(bloque.strip('\n')))
nb = nbformat.v4.new_notebook(cells=celdas, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbformat.write(nb, sys.argv[2]); print(len(celdas), 'celdas')
