const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${CWD}/core/Tutorial.js`);

class TutorialPage extends React.Component {
  render() {
      const {config: siteConfig} = this.props;
      const {baseUrl} = siteConfig;
      return <Tutorial baseUrl={baseUrl} tutorialID="risk_averse_bo_with_environmental_variables"/>;
  }
}

module.exports = TutorialPage;

