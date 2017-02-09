import os
import glob

import nbformat
from nbconvert import RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor


def convert(nbfile):
    basename, _ = os.path.splitext(nbfile)

    meta = {'metadata': {'path': '.'}}
    with open(nbfile, 'r', encoding='utf-8') as nbf:
        nbdata = nbformat.read(nbf, as_version=4, encoding='utf-8')

    runner = ExecutePreprocessor(timeout=600, kernel_name='probscale')
    runner.preprocess(nbdata, meta)

    img_folder = basename + '_files'
    body_raw, images = RSTExporter().from_notebook_node(nbdata)
    body_final = body_raw.replace('.. image:: ', '.. image:: {}/'.format(img_folder))

    with open(basename + '.rst', 'w', encoding='utf-8') as rst_out:
        rst_out.write(body_final)

    for img_name, img_data in images['outputs'].items():
        img_path = os.path.join(img_folder, img_name)
        with open(img_path, 'wb') as img:
            img.write(img_data)


if __name__ == '__main__':
    for nbfile in glob.glob('*.ipynb'):
        convert(nbfile)
