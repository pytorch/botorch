/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from "@theme/Layout";
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import CodeBlock from '@theme/CodeBlock';

const features = [
    {
      content:
        'Plug in new models, acquisition functions, and optimizers.',
      image: 'img/puzzle_pieces.svg',
      title: 'Modular',
    },
    {
      content:
        'Easily integrate neural network modules. Native GPU & autograd support.',
      image: 'img/pytorch_logo.svg',
      title: 'Built on PyTorch',
    },
    {
      content:
        'Support for scalable GPs via GPyTorch. Run code on multiple devices.',
      image: 'img/expanding_arrows.svg',
      title: 'Scalable',
    },
];

const Feature = ({imageUrl, title, content, image}) => {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className='col col--4 feature text--center'>
      {imgUrl && (
        <div>
          <img src={imgUrl} alt={title} />
        </div>
      )}
      {image && (
        <div>
          <img
            className="margin--md"
            src={image}
            alt={title}
            style={{width: '80px', height: '80px'}}
          />
        </div>
      )}
      <h2>{title}</h2>
      <p>{content}</p>
    </div>
  );
}

const HomeSplash = () => {
  const {siteConfig} = useDocusaurusContext();
  const language = 'en';
  const docsUrl = "docs"
  const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
  const langPart = `${language ? `${language}/` : ''}`;
  const docUrl = doc => `${docsPart}${langPart}${doc}`;

  return (
    // <SplashContainer>
    //   <Logo img_src={useBaseUrl('img/botorch_logo_lockup_top.png')} />
    //   <div className="inner">
    //     <ProjectTitle siteConfig={siteConfig} />
    //     <PromoSection>
    //       <Button href={docUrl('introduction.html')}>Introduction</Button>
    //       <Button href={'#quickstart'}>Get Started</Button>
    //       <Button href={useBaseUrl('tutorials/')}>Tutorials</Button>
    //     </PromoSection>
    //   </div>
    // </SplashContainer>
    <div className="homeContainer text--center" style={{ height: "30rem" }}>
      <div className="container">
        <div className="padding-vert--md">
          <img src={useBaseUrl('img/botorch_logo_lockup_top.png')} alt="Project Logo" style={{ width: "300px" }} />
          <p className="hero__subtitle text--secondary">{siteConfig.tagline}</p>
        </div>
        <div>
          <Link
            to="/docs/introduction"
            className="button button--lg button--outline button--secondary margin--sm">
            Introduction
          </Link>
          <Link
            to="#quickstart"
            className="button button--lg button--outline button--secondary margin--sm">
            Get started
          </Link>
          <Link
            to="/docs/tutorials/"
            className="button button--lg button--outline button--secondary margin--sm">
            Tutorials
          </Link>
        </div>
      </div>
    </div>
  );
}

export default () => {
  const {siteConfig} = useDocusaurusContext();
  const language = 'en';

  const Block = props => (
    <div
      padding={['bottom', 'top']}
      id={props.id}
      background={props.background}>
      <div
        align="center"
        contents={props.children}
        layout={props.layout}
      />
    </div>
  );

  const Description = () => (
    <Block background="light">
      {[
        {
          content:
            'This is another description of how this project is useful',
          image: useBaseUrl('img/botorch_logo_lockup_white.svg'),
          imageAlign: 'right',
          title: 'Description',
        },
      ]}
    </Block>
  );
  // getStartedSection
  // Example for model fitting
  const modelFitCodeExample = `import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2, dtype=torch.double) * 2
Y = 1 - torch.linalg.norm(train_X - 0.5, dim=-1, keepdim=True)
Y = Y + 0.1 * torch.randn_like(Y)  # add some noise

gp = SingleTaskGP(
  train_X=train_X,
  train_Y=Y,
  input_transform=Normalize(d=2),
  outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)`;
  // Example for defining an acquisition function
  const constrAcqFuncExample = `from botorch.acquisition import LogExpectedImprovement

logEI = LogExpectedImprovement(model=gp, best_f=Y.max())`;
  // Example for optimizing candidates
  const optAcqFuncExample = `from botorch.optim import optimize_acqf

bounds = torch.stack([torch.zeros(2), torch.ones(2)]).to(torch.double)
candidate, acq_value = optimize_acqf(
  logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)
candidate  # tensor([[0.2981, 0.2401]], dtype=torch.float64)`;
  const papertitle = `BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization`
  const paper_bibtex = `@inproceedings{balandat2020botorch,
  title = {{BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization}},
  author = {Balandat, Maximilian and Karrer, Brian and Jiang, Daniel R. and Daulton, Samuel and Letham, Benjamin and Wilson, Andrew Gordon and Bakshy, Eytan},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year = 2020,
  url = {http://arxiv.org/abs/1910.06403}
}`;
  //
  const QuickStart = () => (
    <div
      className="padding--xl"
      id="quickstart"
      style={{}}>
      <h2 className='text--center padding--md'>Get Started</h2>
      <div>
        <ol>
          <li>
            <h4>Install BoTorch:</h4>
            via pip (recommended):
            <CodeBlock language="bash">pip install botorch</CodeBlock>
            via Anaconda (from the unofficial conda-forge channel):
            <CodeBlock language="bash">conda install botorch -c gpytorch -c conda-forge</CodeBlock>
          </li>
          <li>
            <h4>Fit a model:</h4>
            <CodeBlock language="python" >{modelFitCodeExample}</CodeBlock>
          </li>
          <li>
            <h4>Construct an acquisition function:</h4>
            <CodeBlock language="python" >{constrAcqFuncExample}</CodeBlock>
          </li>
          <li>
            <h4>Optimize the acquisition function:</h4>
            <CodeBlock language="python" >{optAcqFuncExample}</CodeBlock>
          </li>
        </ol>
      </div>
    </div>
  );

  const Reference = () => (
    <div
      className="padding--lg"
      id="reference"
      style={{}}>
      <h2 className='text--center'>References</h2>
      <div>
        <a href={`https://arxiv.org/abs/1910.06403`}>{papertitle}</a>
        <CodeBlock className='margin-vert--md'>{paper_bibtex}</CodeBlock>
        Check out some <a href={`/docs/papers`}>other papers using BoTorch</a>.
      </div>
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

    const pageUrl = page => (language ? `${language}/` : '') + page;

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
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <HomeSplash siteConfig={siteConfig} language={language} />
      <div className="landingPage mainContainer">
        <div className="padding--xl">
        <h2 className="text--center padding--md">Key Features</h2>
        {features && features.length > 0 && (
          <div className="row">
            {features.map(({title, imageUrl, content, image}) => (
              <Feature
                key={title}
                title={title}
                imageUrl={imageUrl}
                content={content}
                image={image}
              />
            ))}
          </div>
        )}
      </div>
        <Reference />
        <QuickStart />
      </div>
    </Layout>
  );
}
