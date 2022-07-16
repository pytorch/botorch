/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary');

const Container = CompLibrary.Container;

const CWD = process.cwd();

const versions = require(`${CWD}/_versions.json`);
versions.sort().reverse();

function Versions(props) {
  const {config: siteConfig} = props;
  const baseUrl = siteConfig.baseUrl;
  const latestVersion = versions[0];
  return (
    <div className="docMainWrapper wrapper">
      <Container className="mainContainer versionsContainer">
        <div className="post">
          <header className="postHeader">
            <h1>{siteConfig.title} Versions</h1>
          </header>

          <table className="versions">
            <tbody>
              <tr>
                <th>Version</th>
                <th>Install with</th>
                <th>Documentation</th>
              </tr>
              <tr>
                <td>{`stable (${latestVersion})`}</td>
                <td>
                  <code>conda install botorch -c pytorch -c gpytorch</code>
                </td>
                <td>
                  <a href={`${baseUrl}index.html`}>stable</a>
                </td>
              </tr>
              <tr>
                <td>
                  {'latest'}
                  {' (main)'}
                </td>
                <td>
                  <code>
                    pip install git+https://github.com/pytorch/botorch.git
                  </code>
                </td>
                <td>
                  <a href={`${baseUrl}v/latest/index.html`}>latest</a>
                </td>
              </tr>
            </tbody>
          </table>

          <h3 id="archive">Past Versions</h3>
          <table className="versions">
            <tbody>
              {versions.map(
                version =>
                  version !== latestVersion && (
                    <tr key={version}>
                      <th>{version}</th>
                      <td>
                        <a href={`${baseUrl}v/${version}/index.html`}>
                          Documentation
                        </a>
                      </td>
                    </tr>
                  ),
              )}
            </tbody>
          </table>
        </div>
      </Container>
    </div>
  );
}

module.exports = Versions;
