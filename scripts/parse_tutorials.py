#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter, PythonExporter


TEMPLATE = """const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${{CWD}}/core/Tutorial.js`);

class TutorialPage extends React.Component {{
  render() {{
      const {{config: siteConfig}} = this.props;
      const {{baseUrl}} = siteConfig;
      return <Tutorial baseUrl={{baseUrl}} tutorialID="{}"/>;
  }}
}}

module.exports = TutorialPage;

"""

JS_SCRIPTS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
"""  # noqa: E501


def validate_tutorial_links(repo_dir: str) -> None:
    """Checks that all .ipynb files that present are linked on the website, and vice
    versa, that any linked tutorial has an associated .ipynb file present.
    """
    with open(os.path.join(repo_dir, "website", "tutorials.json"), "r") as infile:
        tutorial_config = json.load(infile)

    tutorial_ids = {x["id"] for v in tutorial_config.values() for x in v}

    tutorials_nbs = {
        fn.replace(".ipynb", "")
        for fn in os.listdir(os.path.join(repo_dir, "tutorials"))
        if fn[-6:] == ".ipynb"
    }

    missing_files = tutorial_ids - tutorials_nbs
    missing_ids = tutorials_nbs - tutorial_ids

    if missing_files:
        raise RuntimeError(
            "The following tutorials are linked on the website, but missing an "
            f"associated .ipynb file: {missing_files}."
        )

    if missing_ids:
        raise RuntimeError(
            "The following tutorial files are present, but are not linked on the "
            "website: {}.".format(", ".join([nbid + ".ipynb" for nbid in missing_ids]))
        )


def gen_tutorials(repo_dir: str) -> None:
    """Generate HTML tutorials for botorch Docusaurus site from Jupyter notebooks.

    Also create ipynb and py versions of tutorial in Docusaurus site for
    download.
    """
    with open(os.path.join(repo_dir, "website", "tutorials.json"), "r") as infile:
        tutorial_config = json.load(infile)

    # create output directories if necessary
    html_out_dir = os.path.join(repo_dir, "website", "_tutorials")
    files_out_dir = os.path.join(repo_dir, "website", "static", "files")
    for d in (html_out_dir, files_out_dir):
        if not os.path.exists(d):
            os.makedirs(d)

    tutorial_ids = {x["id"] for v in tutorial_config.values() for x in v}

    for tid in tutorial_ids:
        print(f"Generating {tid} tutorial")

        # convert notebook to HTML
        ipynb_in_path = os.path.join(repo_dir, "tutorials", f"{tid}.ipynb")
        with open(ipynb_in_path, "r") as infile:
            nb_str = infile.read()
            nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)

        # displayname is absent from notebook metadata
        nb["metadata"]["kernelspec"]["display_name"] = "python3"

        exporter = HTMLExporter(template_name="classic")
        html, meta = exporter.from_notebook_node(nb)

        # pull out html div for notebook
        soup = BeautifulSoup(html, "html.parser")
        nb_meat = soup.find("div", {"id": "notebook-container"})
        del nb_meat.attrs["id"]
        nb_meat.attrs["class"] = ["notebook"]
        html_out = JS_SCRIPTS + str(nb_meat)

        # generate html file
        html_out_path = os.path.join(
            html_out_dir,
            f"{tid}.html",
        )
        with open(html_out_path, "w") as html_outfile:
            html_outfile.write(html_out)

        # generate JS file
        script = TEMPLATE.format(tid)
        js_out_path = os.path.join(
            repo_dir, "website", "pages", "tutorials", f"{tid}.js"
        )
        with open(js_out_path, "w") as js_outfile:
            js_outfile.write(script)

        # output tutorial in both ipynb & py form
        ipynb_out_path = os.path.join(files_out_dir, f"{tid}.ipynb")
        with open(ipynb_out_path, "w") as ipynb_outfile:
            ipynb_outfile.write(nb_str)
        exporter = PythonExporter()
        script, meta = exporter.from_notebook_node(nb)
        # make sure to use python3 shebang
        script = script.replace("#!/usr/bin/env python", "#!/usr/bin/env python3")
        py_out_path = os.path.join(repo_dir, "website", "static", "files", f"{tid}.py")
        with open(py_out_path, "w") as py_outfile:
            py_outfile.write(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JS, HTML, ipynb, and py files for tutorials."
    )
    parser.add_argument(
        "-w",
        "--repo_dir",
        metavar="path",
        required=True,
        help="botorch repo directory.",
    )
    args = parser.parse_args()
    validate_tutorial_links(args.repo_dir)
    gen_tutorials(args.repo_dir)
