/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null}/>
          <Container className="mainContainer documentContainer postContainer">
                <div className="post">
                  <header className="postHeader">
                    <h1 className="postHeaderTitle">Welcome to botorch's tutorials</h1>
                  </header>
                </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
