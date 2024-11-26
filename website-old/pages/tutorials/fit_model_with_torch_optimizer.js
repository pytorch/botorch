const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${CWD}/core/Tutorial.js`);

class TutorialPage extends React.Component {
  render() {
      const {config: siteConfig} = this.props;
      const {baseUrl} = siteConfig;
      return <Tutorial baseUrl={baseUrl} tutorialID="fit_model_with_torch_optimizer"/>;
  }
}

module.exports = TutorialPage;

