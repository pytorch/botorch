/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock;
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="projectLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        {siteConfig.title}
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href={docUrl('introduction.html')}>Introduction</Button>
            <Button href={"#quickstart"}>Get Started</Button>
            <Button href={`${baseUrl}tutorials/`}>Tutorials</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const Description = () => (
      <Block background="light">
        {[
          {
            content:
              'This is another description of how this project is useful',
            image: `${baseUrl}img/botorch_logo_lockup.svg`,
            imageAlign: 'right',
            title: 'Description',
          },
        ]}
      </Block>
    );
    // getStartedSection
    const pre = "```";
    // Example for model fitting
    const modelFitCodeExample = `${pre}python
>>> import torch
>>> from botorch.models import SingleTaskGP
>>> from botorch.fit import fit_gpytorch_model
>>> from gpytorch.mlls import ExactMarginalLogLikelihood

>>> train_X = torch.rand(10, 2)
>>> Y = 1 - torch.norm(train_X - 0.5, dim=-1) + 0.1 * torch.rand(10)
>>> train_Y = (Y - Y.mean()) / Y.std()

>>> gp = SingleTaskGP(train_X, train_Y)
>>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
>>> fit_gpytorch_model(mll);
    `;
    // Example for defining an acquisition function
    const constrAcqFuncExample = `${pre}python
>>> from botorch.acquisition import UpperConfidenceBound

>>> UCB = UpperConfidenceBound(gp, beta=0.1)
    `;
    // Example for optimizing candidates
    const optAcqFuncExample = `${pre}python
>>> from botorch.optim import joint_optimize

>>> bounds = torch.stack([torch.zeros(2), torch.ones(2)])
>>> candidate = joint_optimize(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
>>> candidate
tensor([0.4887, 0.5063])
    `;
    //
    const QuickStart = () => (
      <div
        className="productShowcaseSection" id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
              <ol>
                <li>
                  Install BoTorch:
                  <MarkdownBlock>{bash`pip install botorch`}</MarkdownBlock>
                </li>
                <li>
                  Fit a model:
                  <MarkdownBlock>{modelFitCodeExample}</MarkdownBlock>
                </li>
                <li>
                  Construct an acquisition function:
                  <MarkdownBlock>{constrAcqFuncExample}</MarkdownBlock>
                </li>
                <li>
                  Optimize the acqusiton function:
                  <MarkdownBlock>{optAcqFuncExample}</MarkdownBlock>
                </li>
              </ol>
          </Container>
        </div>
    );

    const Features = () => (
      <div
        className="productShowcaseSection"
        style={{textAlign: 'center'}}>
        <h2>Key Features</h2>
      <Block layout="threeColumn">
        {[
          {
            content: 'Plug in new models, acquisition functions, and optimizers.',
            image: `${baseUrl}img/th-large-solid.svg`,
            imageAlign: 'top',
            title: 'Modular',
          },
          {
            content: 'Easily integrate neural network modules. Native GPU & autograd support.',
            image: `${baseUrl}img/pytorch-logo.png`,
            imageAlign: 'top',
            title: 'Built on PyTorch',
          },
          {
            content: 'Support for scalable GPs. Run code on multiple devices.',
            image: `${baseUrl}img/expand-arrows-alt-solid.svg`,
            imageAlign: 'top',
            title: 'Scalable',
          },
        ]}
      </Block>
      </div>
    );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
