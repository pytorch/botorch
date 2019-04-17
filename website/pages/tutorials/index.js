/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
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
                    <h1 className="postHeaderTitle">Welcome to the botorch tutorials</h1>
                  </header>
	          <body>
	          <p>botorch (pronounced like blow-torch) is a library for Bayesian Optimization research 
	          built on top of PyTorch, and is part of the PyTorch ecosystem.</p>
	          <p>Botorch is best used in tandem with Ax, Facebook's open-source adaptive experimentation platform, 
	          which provides an easy-to-use Bayesian optimization interface while handling various 
	          experiment and data management, transformations, and systems integration.</p>
	          
	    	  <p>The tutorials here will help you undertand and use botorch in your own work.</p>
	    	  <bl>
	    		<li><a href="custom_botorch_model_in_ax">Using a custom botorch model</a></li>
	    		<li><a href="fit_model_with_torch_optimizer">Fitting a model using torch.optim</a></li>
			<li><a href="compare_mc_analytic_acquisition">Comparing analytic and MC Expected Improvement</a></li>
                      	<li><a href="batch_mode_cross_validation">Using batch evaluation for fast cross-validation</a></li>
	                <li><a href="custom_acquisition">Writing a custom acquisition function and interfacing with Ax</a></li>
	          </bl>
	    	  
	          </body>
	          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
