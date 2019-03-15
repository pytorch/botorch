/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const fs = require('fs-extra');
const path = require('path');
const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class Tutorial extends React.Component {
  render() {
    const {tutorialID} = this.props;

    const htmlFile = `${CWD}/_tutorials/${tutorialID}.html`;
    const normalizedHtmlFile = path.normalize(htmlFile);

    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={tutorialID} />
        <Container className="mainContainer">
          <div
            className="tutorialBody"
            dangerouslySetInnerHTML={{
              __html: fs.readFileSync(normalizedHtmlFile, {encoding: 'utf8'}),
            }}
          />
          <div className="tutorialButtonWrapper buttonWrapper">
            <a
              className="tutorialButton button"
              href={`/files/${tutorialID}.ipynb`}>
              <i className="fas fa-file-download" />
              <img
                src={'/img/file-download-solid.svg'}
                height="15"
                width="15"
              />
              {'Download Tutorial Jupyter Notebook'}
            </a>
          </div>
          <div className="tutorialButtonWrapper buttonWrapper">
            <a
              className="tutorialButton button"
              href={`/files/${tutorialID}.py`}>
              <img
                src={'/img/file-download-solid.svg'}
                height="15"
                width="15"
              />
              {'Download Tutorial Source Code'}
            </a>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = Tutorial;
