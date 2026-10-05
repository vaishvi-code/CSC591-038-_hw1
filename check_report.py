import re, os
text = open('report.tex', encoding='utf-8').read()
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs   = set(re.findall(r'\\ref\{([^}]+)\}', text))
missing = refs - labels
print('Labels defined:', sorted(labels))
print('Refs used:     ', sorted(refs))
print('Missing labels:', missing if missing else 'NONE')
imgs = re.findall(r'\\includegraphics\[.*?\]\{([^}]+)\}', text)
print('\nImage files:')
for img in imgs:
    status = 'OK' if os.path.exists(img) else 'MISSING'
    print(f'  {status}: {img}')
