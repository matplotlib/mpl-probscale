import os
import glob

import nbformat
from nbconvert import RSTExporter, preprocessors


def cache(nbfile):
    basename, _ = os.path.splitext(nbfile)
    bakfile = basename + '.bak'
    with open(nbfile, 'r') as nb, open(bakfile, 'w') as bak:
        bak.write(nb.read())
    return bakfile


def process(nbfile, processor):
    meta = {'metadata': {'path': '.'}}
    with open(nbfile, 'r') as nbf:
        nbook = nbformat.read(nbf, as_version=4)

    runner = processor(timeout=600, kernel_name='probscale')
    runner.preprocess(nbook, meta)

    with open(nbfile, 'w') as nbf:
        nbformat.write(nbook, nbf)


def convert(nbfile):
    basename, _ = os.path.splitext(nbfile)
    img_folder = basename + '_files'
    os.makedirs(img_folder, exist_ok=True)
    print("\tconverting " + nbfile)

    with open(nbfile, 'r') as nb:
        nbdata = nbformat.reads(nb.read(), as_version=4)

    rst = RSTExporter()
    body_raw, images = rst.from_notebook_node(nbdata)
    body_final = body_raw.replace('.. image:: ', '.. image:: {}/'.format(img_folder))

    with open(basename + '.rst', 'w') as rst_out:
        rst_out.write(body_final)

    for img_name, img_data in images['outputs'].items():
        img_path = os.path.join(img_folder, img_name)
        with open(img_path, 'wb') as img:
            print('\twriting' + img_path)
            img.write(img_data)


if __name__ == '__main__':
    for nbfile in glob.glob('*.ipynb'):
        bak = cache(nbfile)
        success = False
        try:
            process(nbfile, preprocessors.ExecutePreprocessor)
            convert(nbfile)
            success = True
        finally:
            process(nbfile, preprocessors.ClearOutputPreprocessor)
        if success:
            os.remove(bak)
